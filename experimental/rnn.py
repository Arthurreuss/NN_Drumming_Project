import torch
import torch.nn as nn


# --- Simple RNN model ---
class DrumRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)         # out: [batch, seq, hidden]
        out = self.fc(out)           # project to instrument dimension
        return torch.sigmoid(out)    # output in [0,1]


