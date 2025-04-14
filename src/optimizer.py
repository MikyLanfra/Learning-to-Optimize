import torch
from torch import nn
import numpy as np

class LSTMConcurrent(nn.Module):
    """
    LSTM-based optimizer as described in the paper.
    """
    def __init__(self, num_optims, hidden_size=20, preproc=True, preproc_factor=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.preproc = preproc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preproc_factor = torch.tensor(preproc_factor, device=self.device)
        self.preproc_threshold = float(torch.exp(-self.preproc_factor))
        
        self.input_size = 2 * num_optims if preproc else 1 * num_optims
        self.lstm = nn.LSTM(self.input_size, hidden_size, 2, batch_first=True).to(self.device)
        self.output_layer = nn.Linear(hidden_size, 1).to(self.device)


    def forward(self, x, hidden_state):
        """
        x: (num_params, 1, input_size)
        hidden_state: tuple of (h, c) with shape (num_layers, num_params, hidden_size)
        """
        x = x.to(self.device)
        if self.preproc:
            x = self.preprocess_gradients(x)  # shape: (num_params, input_size)
        
        x = x.unsqueeze(1)  # (num_params, 1, input_size) to match LSTM's (batch, seq_len, input_size)

        out, new_hidden_state = self.lstm(x, hidden_state)  # Efficient batch LSTM call
        out = self.output_layer(out).squeeze(1)  # (num_params, 1, 1) → (num_params, 1)
        return out, new_hidden_state

    def preprocess_gradients(self, gradients):
        """ Applies log transformation & sign extraction to gradients. """
        gradients = gradients.data.to(self.device)
        if len(gradients.size()) == 1:
            gradients = gradients.unsqueeze(-1)

        param_size = gradients.size(0)
        num_optims = gradients.size(1)

        preprocessed = torch.zeros(param_size, 2 * num_optims, device=self.device)

        for i in range(num_optims):
            gradient = gradients[:, i]
            keep_grads = (torch.abs(gradient) >= self.preproc_threshold)

            # Log transformation for large gradients
            preprocessed[keep_grads, 2 * i] = (torch.log(torch.abs(gradient[keep_grads]) + 1e-8) / self.preproc_factor)
            preprocessed[keep_grads, 2 * i + 1] = torch.sign(gradient[keep_grads])

            # Direct scaling for small gradients
            preprocessed[~keep_grads, 2 * i] = -1
            preprocessed[~keep_grads, 2 * i + 1] = (float(torch.exp(self.preproc_factor)) * gradient[~keep_grads])

        return preprocessed


    def initialize_hidden_state(self, num_params):
        h0 = torch.zeros(2, num_params, self.hidden_size, device=self.device)
        c0 = torch.zeros(2, num_params, self.hidden_size, device=self.device)
        return (h0, c0)
