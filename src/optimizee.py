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
        pass

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