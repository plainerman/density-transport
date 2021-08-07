import pyro.distributions as dist
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


def simulate_bimodal():
    dist_x1 = dist.Normal(torch.tensor([-3.0]), torch.tensor([2.0]))
    dist_x2 = dist.Normal(torch.tensor([3.0]), torch.tensor([1.0]))

    x = np.concatenate([dist_x1.sample([50000]).numpy(), dist_x2.sample([50000]).numpy()])

    # We center it, so learning is easier
    return StandardScaler().fit_transform(x)


def simulate_trimodal():
    dist_x1 = dist.Normal(torch.tensor([-3.0]), torch.tensor([1.0]))
    dist_x2 = dist.Normal(torch.tensor([4.5]), torch.tensor([2.0]))
    dist_x3 = dist.Normal(torch.tensor([-0.5]), torch.tensor([0.5]))

    x = np.concatenate([dist_x1.sample([50000]).numpy(), dist_x2.sample([50000]).numpy(), dist_x3.sample([50000]).numpy()])

    return StandardScaler().fit_transform(x)
