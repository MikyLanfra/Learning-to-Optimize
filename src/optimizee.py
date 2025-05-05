import torch
import torch.optim as optim
import numpy as np
from torch import nn


class Optimizee:
    """
    Abstract class for optimization problems.
    """
    def __init__(self):
        """
        Initialize the optimization problem.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_params(self, params=None):
        """
        Returns initial parameters for optimization.
        
        Args:
            params (torch.Tensor of shape (d,1) or None): Parameters setting. If None, random initialization
        
        Returns:
            torch.Tensor: Initialized parameters.
        """
        raise NotImplementedError("Subclasses must implement set_params.")
    
    def compute_loss(self, params):
        """
        Computes the loss function value.
        
        Args:
            params (torch.Tensor): Current parameter values.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("Subclasses must implement compute_loss.")
    
    def all_parameters(self):
        """
        Returns all parameters of the optimizee, as a tensor of shape (d,1).

        Returns:
            torch.Tensor of shape (d,1).       
        """
        raise NotImplementedError("Subclasses must implement all_parameters.")
    
    def train(self):
        """
        Allows for setting Model to training mode.
        """
        pass
    
    def eval(self):
        """
        Allows for setting Model to training mode.
        """
        pass

class QuadraticOptimizee(Optimizee):
    """
    Class for the quadratic function described in the paper.
    """
    def __init__(self, W, theta0, noise_std=0.01):
        """
        Initialize the quadratic function.

        Args:
            W (np.ndarray): 10x10 matrix.
            theta0 (np.ndarray): 10x1 vector (true parameters).
            noise_std (float): Standard deviation of the noise term.
        """
        super().__init__()
        self.W = torch.tensor(W, dtype=torch.float32)
        self.theta0 = torch.tensor(theta0, dtype=torch.float32)
        self.noise_std = noise_std
        self.theta = None

        # Generate noisy observations y = W @ theta0 + eps
        self.y = self.W @ self.theta0 + self.noise_std * torch.randn_like(self.theta0)

    def set_params(self, params=None):
        """
        Returns initial parameters for optimization (random initialization).
        """
        self.theta = torch.randn_like(self.theta0, requires_grad=True) if params is None else params

    def compute_loss(self, params, return_grad=True):
        """
        Computes the loss ||W @ params - y||^2.
        """
        if return_grad:
            loss = torch.norm((self.W.matmul(params) - self.y) ** 2)
            grads = torch.autograd.grad(loss, params, create_graph=True)[0]
            detached_grads = torch.tensor(grads.detach().numpy(), requires_grad=True)
            return loss, detached_grads
        else:
            return torch.norm((self.W.matmul(params) - self.y) ** 2)
    
    def all_parameters(self):
        """
        Returns all parameters of the optimizee, as a tensor of shape (d,1).
        """
        return self.theta
    
class XYNNOptimizee(Optimizee):
    """
    Class for a generic linear neural network optimizee.
    """
    def __init__(self, X, y, hidden_size=20, num_layers=2, num_samples=10, loss_fn=nn.MSELoss(), activation_fn=nn.ReLU()):
        """
        """
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device)

        self.input_size = X.shape[1]
        self.hidden_size = hidden_size
        try:
          self.output_size = y.shape[1]
        except:
          self.output_size = 1

        self.num_layers = num_layers
        self.num_samples = num_samples
        self.loss_fn = loss_fn.to(self.device)
        self.activation_fn = activation_fn.to(self.device)
        self.model = []
        for i in range(num_layers):
            if num_layers == 1:
                W = torch.randn(self.input_size+1, self.output_size, requires_grad=True, device=self.device)
            elif i == 0:
                W = torch.randn(self.input_size+1, self.hidden_size, requires_grad=True, device=self.device)
            elif i == num_layers - 1:
                W = torch.randn(self.hidden_size+1, self.output_size, requires_grad=True, device=self.device)
            else:
                W = torch.randn(self.hidden_size+1, self.hidden_size, requires_grad=True, device=self.device)
            self.model.append(W)
    
    
    def set_params(self, params=None):
        """
        Given a tensor of shape (d,1), sets the parameters of the optimizee.
        """
        if params is not None:
            params = torch.flatten(params).to(self.device)
            for i in range(self.num_layers):
                current_param = self.model[i]
                in_size, out_size = current_param.shape
                new_param = params[:in_size*out_size].reshape(in_size, out_size)
                params = params[in_size*out_size:]
                self.model[i] = new_param.to(self.device)

    def forward(self, x):
        """
        Forward pass of the linear neural network.
        """
        x = x.to(self.device).flatten(start_dim=1)  # Flatten the input
        ones = torch.ones(x.shape[0], 1, device=self.device)  # Create a column of ones
        for i in range(self.num_layers):
            W = self.model[i]
            x = torch.cat((x, ones), dim=1)
            x = x @ W
            x = self.activation_fn(x) if i < self.num_layers - 1 else x
        return x

    def compute_loss(self, params, return_grad=True):
        self.set_params(params)  # Set model parameters
        total_loss = None
        
        indices = torch.randint(0, self.X.shape[0], (self.num_samples,), device=self.X.device)
        inputs = self.X[indices]
        targets = self.y[indices].squeeze()

        outputs = self.forward(inputs).squeeze()

        total_loss = self.loss_fn(outputs, targets)

        if return_grad:
            grads = torch.autograd.grad(total_loss, self.model, create_graph=True)
            grads = torch.cat([g.flatten() for g in grads]).unsqueeze(-1)
            return total_loss, grads.detach()
        else:
            return total_loss

    
    def all_parameters(self):
        """
        Returns all parameters of the optimizee, as a tensor of shape (d,1).
        """
        return torch.cat([p.flatten() for p in self.model]).unsqueeze(-1).to(self.device)
    
    def oh_labels(self):
        return self.y.sum(dim=0)>0