import torch
from torch import nn
from tqdm.notebook import tqdm


def initialize_optimizees(optimizee_cls, optimizee_kwargs, num_optimizees=10, noise='equal'):
    optimizees = []
    for i in range(num_optimizees):
        if noise == 'equal':
            optim = optimizee_cls(**optimizee_kwargs)
            optim.train()
            optimizees.append(optim)
        else:
            optimizee_kwargs['noise_std'] = 0.01 * (i+1)
            optim = optimizee_cls(**optimizee_kwargs)
            optim.train()
            optimizees.append(optim)
    return optimizees




def train_LSTM(lstm_optimizer, meta_optimizer, optimizee_class, optimizee_kwargs, num_optimizees=1, num_epochs=500, time_horizon=200, discount=1, scheduler = None, noise='equal', writer=None):
    lstm_optimizer.train()
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ConstantLR(meta_optimizer, factor=1.0, total_iters=num_epochs)

    
    with tqdm(range(num_epochs), desc="Training Progress") as pbar:
        for epoch in pbar:
            # Initialize optimizee parameters
            optimizees = initialize_optimizees(optimizee_class, optimizee_kwargs, num_optimizees, noise=noise)
            optimizees[0].set_params()
            params = optimizees[0].all_parameters()
            
            cumulative_loss = None             
            for t in range(time_horizon):
                gradients = []
                hidden_state = lstm_optimizer.initialize_hidden_state()
                for i in range(num_optimizees):
                    optimizee = optimizees[i]
                    loss, grad_params = optimizee.compute_loss(params, return_grad=True)
                    if i == 0 and discount: cumulative_loss = loss*discount**(time_horizon-1) if cumulative_loss is None else cumulative_loss + loss*discount**(time_horizon-t-1)
                    elif i==0: cumulative_loss = loss
                    gradients.append(grad_params.squeeze())
                    if writer and i==0 and epoch==1: writer.add_scalar("Grad", grad_params.squeeze().mean(), t)

                # Stack gradients
                grad_params = torch.stack(gradients).T
                update, hidden_state = lstm_optimizer(grad_params, hidden_state)
                params = params + update
                if writer and epoch==1: writer.add_scalar("Update", update.mean(), t)
                optimizees[0].set_params(params)

            
            # Backpropagation through time (BPTT)
            if writer: writer.add_scalar("Loss", cumulative_loss, epoch)
            meta_optimizer.zero_grad()
            cumulative_loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_optimizer.parameters(), 1)
            meta_optimizer.step()
            scheduler.step()

            # Update progress bar
            pbar.set_postfix(loss=cumulative_loss.item())
            if (epoch + 1) % 1 == 0:
                current_lr = meta_optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}], Cumulative Loss: {cumulative_loss.item():.4f}, LR: {current_lr:.3e}")
                print(f"Final parameters: {params.detach().numpy().T}")
                
    print("\nTraining complete!")
    return lstm_optimizer




def test_LSTM(lstm_optimizer, optimizee_class, optimizee_kwargs, num_optimizees=1, time_horizon=200, noise='equal', writer=None):
    lstm_optimizer.eval()
    optimizees = initialize_optimizees(optimizee_class, optimizee_kwargs, num_optimizees, noise=noise)    
    optimizees[0].set_params()
    params = optimizees[0].all_parameters()
    hidden_state = lstm_optimizer.initialize_hidden_state()
    for t in range(time_horizon):
        gradients = []
        for i in range(num_optimizees):
            optimizee = optimizees[i]
            loss, grad_params = optimizee.compute_loss(params)
            if writer and i==0: writer.add_scalar("Loss", loss, t)
            gradients.append(grad_params.squeeze())
        
        grad_params = torch.stack(gradients).T

        updates, hidden_state = lstm_optimizer(grad_params, hidden_state)
        params = params - updates 
        optimizees[0].set_params(params)

    return params