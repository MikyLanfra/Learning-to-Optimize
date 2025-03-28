import torch
from torch import nn
from tqdm.notebook import tqdm


def initialize_optimizees(optimizee_cls, optimizee_kwargs, num_optimizees=10, noise='equal'):
    optimizees = []
    params_list = []
    for i in range(num_optimizees):
        if noise == 'equal':
            optimizees.append(optimizee_cls(**optimizee_kwargs))
        else:
            optimizee_kwargs['noise_std'] = 0.01 * (i+1)
            optimizees.append(optimizee_cls(**optimizee_kwargs))
        param = optimizees[i].get_initial_params()
        params_list.append(param)
    return optimizees, params_list




def train_LSTM(lstm_optimizer, meta_optimizer, optimizee_class, optimizee_kwargs, num_optimizees=1, num_epochs=500, time_horizon=200, discount=1):
    with tqdm(range(num_epochs), desc="Training Progress") as pbar:
        for epoch in pbar:
            # Initialize optimizee parameters
            optimizees, params_list = initialize_optimizees(optimizee_class, optimizee_kwargs, num_optimizees, noise='not_equal')
            hidden_state = lstm_optimizer.initialize_hidden_state(num_optimizees)

            cumulative_losses = [None] * num_optimizees                
            for t in range(time_horizon):
                gradients = []
                for i in range(num_optimizees):
                    optimizee = optimizees[i]
                    params = params_list[i]
                    cumulative_loss = cumulative_losses[i]
                    
                    loss = optimizee.compute_loss(params)
                    cumulative_losses[i] = loss*discount**(time_horizon) if cumulative_loss is None else cumulative_loss + loss*discount**(time_horizon-t)
                    
                    # Compute gradients of the loss w.r.t. the parameters
                    grad_params = torch.autograd.grad(loss, params, create_graph=True)[0]
                    grad_params = grad_params.detach()

                    gradients.append(grad_params)

                # Stack gradients
                grad_params = torch.stack(gradients)
                # print(grad_params.shape)

                updates, hidden_state = lstm_optimizer(grad_params, hidden_state)
                
                for i in range(num_optimizees):
                    params_list[i] = params_list[i] - updates[i]

                
            # Backpropagation through time (BPTT)
            # cumulative_loss = loss
            cumulative_loss_total = sum(cumulative_losses) / num_optimizees

            meta_optimizer.zero_grad()
            cumulative_loss_total.backward()
            meta_optimizer.step()

            # Update progress bar
            pbar.set_postfix(loss=cumulative_loss.item())
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Cumulative Loss: {cumulative_loss.item():.4f}")
                print(f"Final parameters: {params_list[0].detach().numpy().T}")
                
    print("\nTraining complete!")
    return lstm_optimizer




def test_LSTM(lstm_optimizer, optimizee_cls, optimizee_kwargs, time_horizon=200):
    optimizee = optimizee_cls(**optimizee_kwargs)
    params = optimizee.get_initial_params()
    hidden_state = lstm_optimizer.initialize_hidden_state(1)
    for t in range(time_horizon):
        loss = optimizee.compute_loss(params)

        grad_params = torch.autograd.grad(loss, params, create_graph=True)[0]
        grad_params = grad_params.detach()

        if len(grad_params.shape) == 2:
            grad_params.unsqueeze_(0)

        updates, hidden_state = lstm_optimizer(grad_params, hidden_state)
        updates = torch.mean(updates, dim=0)
        params = params - updates 

    print(f"Final parameters: {params.detach().numpy().T}")
    return params