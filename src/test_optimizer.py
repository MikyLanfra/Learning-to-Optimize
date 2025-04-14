import torch
from tqdm.notebook import tqdm

def test_optimizer(optimizer_cls, optimizee, optimizer_kwargs={}, time_horizon=200, writer=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizee.set_params()
    params = optimizee.all_parameters().to(device)

    optimizer = optimizer_cls([params], **optimizer_kwargs)

    for t in tqdm(range(time_horizon), desc="Training Progress", unit="time step"):
        loss = optimizee.compute_loss(params, return_grad=False).to(device)
        if writer: writer.add_scalar("Loss", loss.item(), t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return params
