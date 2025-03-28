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

    def get_initial_params(self, params=None):
        """
        Returns initial parameters for optimization.
        
        Args:
            params (torch.Tensor or None): Optional initial parameters.
        
        Returns:
            torch.Tensor: Initialized parameters.
        """
        raise NotImplementedError("Subclasses must implement get_initial_params.")
    
    def compute_loss(self, params):
        """
        Computes the loss function value.
        
        Args:
            params (torch.Tensor): Current parameter values.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("Subclasses must implement compute_loss.")
    
    def all_named_parameters(self):
        """
        Returns all named parameters of the optimizee.
        
        Returns:
            list of tuples: List containing (name, parameter) pairs.
        """
        raise NotImplementedError("Subclasses must implement all_named_parameters.")


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

        # Generate noisy observations y = W @ theta0 + eps
        self.y = self.W @ self.theta0 + self.noise_std * torch.randn_like(self.theta0)

    def get_initial_params(self, params=None):
        """
        Returns initial parameters for optimization (random initialization).
        """
        self.theta = torch.randn_like(self.theta0, requires_grad=True) if params is None else params
        return self.theta

    def compute_loss(self, params):
        """
        Computes the loss ||W @ params - y||^2.
        """
        return torch.norm((self.W.matmul(params) - self.y) ** 2)
    
    def all_named_parameters(self):
        """
        Returns all parameters of the optimizee.
        """
        return [('theta', self.theta)]
    


