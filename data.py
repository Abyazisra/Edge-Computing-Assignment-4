import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T

import config


def _tf(name, augment):
    mean, std = config.STATS[name]
    ops = []
    if augment:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def _dataset(name, train, augment, smoke):
    ncls = 100 if name == "cifar100" else 10
    if smoke:
        # tiny random data, only used to check that the scripts run
        size = 256 if train else 128
        return datasets.FakeData(size, (3, 32, 32), ncls, transform=_tf(name, augment))
    cls = datasets.CIFAR100 if name == "cifar100" else datasets.CIFAR10
    return cls(config.DATA, train=train, download=True, transform=_tf(name, augment))


def _loader(ds, batch, shuffle, workers):
    cuda = torch.cuda.is_available()
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=cuda, drop_last=False,
                      persistent_workers=workers > 0)


def get_loaders(name, batch=128, smoke=False, val_size=0, train_subset=0, workers=2, seed=0):
    """Returns (train, val, test). val is None unless val_size > 0."""
    if smoke:
        workers = 0
    train_ds = _dataset(name, True, True, smoke)
    test_ds = _dataset(name, False, False, smoke)
    val_dl = None
    if val_size:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_ds), generator=g).tolist()
        val_idx, tr_idx = perm[:val_size], perm[val_size:]
        if train_subset:
            tr_idx = tr_idx[:train_subset]
        plain = _dataset(name, True, False, smoke)
        val_dl = _loader(Subset(plain, val_idx), batch * 2, False, workers)
        train_ds = Subset(train_ds, tr_idx)
    train_dl = _loader(train_ds, batch, True, workers)
    test_dl = _loader(test_ds, batch * 2, False, workers)
    return train_dl, val_dl, test_dl
