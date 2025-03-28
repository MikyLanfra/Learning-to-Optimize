from torch import nn
import numpy as np
from torch_utils import *


class LSTMConcurrent(nn.Module):
    """
    LSTM-based optimizer as described in the paper.
    """
    def __init__(self, num_optims, hidden_size=20, preproc=True, preproc_factor=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.preproc = preproc
        self.preproc_factor = torch.tensor(preproc_factor)
        self.preproc_threshold = float(torch.exp(self.preproc_factor))
        
        self.input_size = 2*num_optims if preproc else 1*num_optims
        self.lstm = nn.LSTM(self.input_size, hidden_size, 2, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)


    def forward(self, x, hidden_state):
        """
        Forward pass of the LSTM optimizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            hidden_state (tuple): Hidden state of the LSTM (h, c).

        Returns:
            torch.Tensor: Output updates of shape (batch_size, sequence_length, 1).
            tuple: Updated hidden state.
        """
        if self.preproc: x = self.preprocess_gradients(x)

        # print("Preprocess Shape", x.shape)
        out, hidden_state = self.lstm(x, hidden_state)
        out = self.output_layer(out)

        return out, hidden_state



    def preprocess_gradients(self, gradients):
        """ Applies log transformation & sign extraction to gradients, moving to CUDA if available. """

        gradients = gradients.data  # Extract raw gradient data
        if len(gradients.size()) == 2: gradients = gradients.unsqueeze(0)
        
        batch_size = gradients.size(0)
        param_size = gradients.size(1)
        num_optims = gradients.size(2)

        preprocessed = torch.zeros(batch_size, param_size, 2*num_optims)

        for i in range(num_optims):
            gradient = gradients[:,:,i]
            keep_grads = (torch.abs(gradient) >= self.preproc_threshold).squeeze()        
        
            # Log transformation for large gradients
            preprocessed[:,:, i][:, keep_grads] = (torch.log(torch.abs(gradient[:, keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            preprocessed[:,:, i+1][:, keep_grads] = torch.sign(gradient[:, keep_grads]).squeeze()

            # Direct scaling for small gradients
            preprocessed[:,:, i][:, ~keep_grads] = -1
            preprocessed[:,:, i+1][:, ~keep_grads] = (float(torch.exp(self.preproc_factor)) * gradient[:, ~keep_grads]).squeeze()

        return torch.tensor(preprocessed)
    

    def initialize_hidden_state(self, batch_sz):
        # Initialize hidden & cell states for LSTM (one per parameter)
        self.h0 = to_cuda(torch.zeros(2, batch_sz, self.hidden_size))
        self.c0 = to_cuda(torch.zeros(2, batch_sz, self.hidden_size))
        return (self.h0, self.c0)
    