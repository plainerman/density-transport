import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# https://github.com/ldickmanns/masters-practicum-mlcms/blob/13b99fb2b37d1c2d4bd27629b4eaabae14522f57/final-project/utils.py
def training_validator_splitter(dataset, bs=100, validation_split=0.2):
    random_seed = 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=bs, sampler=valid_sampler)

    return train_loader, valid_loader
