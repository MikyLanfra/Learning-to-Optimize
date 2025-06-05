import torch
from tqdm.notebook import tqdm

def train_LSTM_alt_preproc(lstm_optimizer, meta_optimizer, initializer, num_epochs=500, min_horizon=100, max_horizon=200, discount=1, scheduler = None, writer=None):
    
    lstm_optimizer.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ConstantLR(meta_optimizer, factor=1.0, total_iters=num_epochs)


    with tqdm(range(num_epochs), desc="Training Progress") as pbar:
        for epoch in pbar:
            optimizees = initializer.initialize()
            optimizees[0].set_params()
            params = optimizees[0].all_parameters().to(device)
            hidden_state = lstm_optimizer.initialize_hidden_state(params.size(0))
            cumulative_loss = None

            if min_horizon < max_horizon: time_horizon = torch.randint(min_horizon, max_horizon, (1,)).item()
            else: time_horizon = max_horizon
            # print(f"Current time horizon: {current_time_horizon}")
            for t in range(time_horizon):
                gradients = []
                for i in range(len(optimizees)):
                    optimizee = optimizees[i]
                    loss, grad_params = optimizee.compute_loss(params, return_grad=True)
                    if i == 0: 
                        if discount: cumulative_loss = loss*discount**(time_horizon-1) if cumulative_loss is None else cumulative_loss + loss*discount**(time_horizon-t-1)
                        else: cumulative_loss = loss
                        gradients.append(grad_params.squeeze().to(device))
                    else:
                        gradients.append(grad_params.squeeze().to(device) - gradients[0])
                    # if writer and i==0 and epoch==1: writer.add_scalar("Grad", grad_params.squeeze().mean(), t)

                grad_params = torch.stack(gradients).T
                # LSTM gradients
                
                # print(f"Gradients: {grad_params.shape}")
                update, hidden_state = lstm_optimizer(grad_params, hidden_state)
                # print(f"Update: {update.shape}")
                params = params + update
                # if writer and epoch==1: writer.add_scalar("Update", update.mean(), t)
                optimizees[0].set_params(params)

            # Backpropagation through time (BPTT)
            if writer: writer.add_scalar("Loss", cumulative_loss, epoch)
            meta_optimizer.zero_grad()
            cumulative_loss.backward()
            # torch.nn.utils.clip_grad_norm_(lstm_optimizer.parameters(), 1)
            meta_optimizer.step()
            scheduler.step()

            # Update progress bar
            pbar.set_postfix(loss=cumulative_loss.item())
            # print(lstm_optimizer.lstm.weight_ih_l0.grad[:10])
            num_prints = num_epochs // 10
            if (epoch + 1) % num_prints == 0:
                current_lr = meta_optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}], Cumulative Loss: {cumulative_loss.item():.4f}, LR: {current_lr:.3e}")
                # print(f"Final parameters: {(params.detach().cpu().numpy().T)[:10]}...")
                

    print("\nTraining complete!")
    return lstm_optimizer




def test_LSTM_alt_preproc(lstm_optimizer, initializer, time_horizon=200, writer=None):
    lstm_optimizer.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizees = initializer.initialize()
    optimizees[0].set_params()
    params = optimizees[0].all_parameters().to(device)
    hidden_state = lstm_optimizer.initialize_hidden_state(params.size(0))

    for t in range(time_horizon):
        gradients = []
        for i in range(len(optimizees)):
            optimizee = optimizees[i]
            loss, grad_params = optimizee.compute_loss(params)
            if writer and i==0: writer.add_scalar("Loss", loss, t)
            if i == 0: 
                gradients.append(grad_params.squeeze().to(device))
            else:
                gradients.append(grad_params.squeeze().to(device) - gradients[0])

        grad_params = torch.stack(gradients).T
        # if len(grad_params.shape)==1: grad_params = grad_params.unsqueeze(-1)

        updates, hidden_state = lstm_optimizer(grad_params, hidden_state)
        params = params + updates
        optimizees[0].set_params(params)

    final_loss = optimizees[0].compute_loss(params, return_grad=False)
    print(f"Final Loss: {final_loss}")
    # print(f"Final parameters: {(params.detach().cpu().numpy().T)[:10]}...")
    return params 