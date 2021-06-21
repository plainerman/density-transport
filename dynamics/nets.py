# This file was (partially) taken from:
# https://github.com/ldickmanns/masters-practicum-mlcms/blob/13b99fb2b37d1c2d4bd27629b4eaabae14522f57/final-project/nets.py
import torch
import torch.nn as nn


class RK4N(nn.Module):
    def __init__(self, input_size=2, num_param=1, hidden_size=20, h=1, num_hidden_layers=2):
        super(RK4N, self).__init__()

        if num_hidden_layers < 1:
            print('Invalid number of specified hidden layers!')
            return

        self.h = h
        self.num_hidden_layers = num_hidden_layers

        # Input.
        self.input = nn.Linear(input_size + num_param, hidden_size, bias=True)

        # Hidden layers.
        self.hidden_layers = {}
        for i in range(num_hidden_layers - 1):
            name = 'h' + str(i)
            self.hidden_layers[name] = nn.Linear(hidden_size, hidden_size, bias=True)

        # Output.
        self.output = nn.Linear(hidden_size, input_size, bias=True)

    def forward(self, x, p):
        # We use the same neural net four times
        k1 = self.one_step(torch.cat((x, p), 1))
        k2 = self.one_step(torch.cat((x + 0.5 * k1, p), 1))
        k3 = self.one_step(torch.cat((x + 0.5 * k2, p), 1))
        k4 = self.one_step(torch.cat((x + k3, p), 1))
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def one_step(self, i):
        f = self.input(i)

        # Hidden layers.
        for i in range(self.num_hidden_layers - 1):
            name = 'h' + str(i)
            f = self.hidden_layers[name](torch.relu(f))  # TODO: what activation function?

        # Output.
        return self.h * self.output(torch.relu(f))  # TODO: same here
