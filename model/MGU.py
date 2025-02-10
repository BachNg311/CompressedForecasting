import torch
import torch.nn as nn

class MGU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MGU, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        # x: input at current time step (batch_size, input_size)
        # h_prev: hidden state at previous time step (batch_size, hidden_size)

        r = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))  # (batch_size, hidden_size)
        h_hat = torch.tanh(self.W(x) + r * self.U(h_prev))  # (batch_size, hidden_size)
        h = (1 - r) * h_prev + r * h_hat  # (batch_size, hidden_size)

        return h