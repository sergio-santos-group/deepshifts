import torch.utils.data
import numpy as np
import torch
import os


def equalize_indices(ys, nbins):
    ymin, ymax = ys.min(), ys.max()
    bins = np.linspace(ymin, ymax, nbins+1)
    bins[-1] += np.inf

    pots = np.digitize(ys, bins)
    bincounts = np.bincount(pots)
    max_count = bincounts.max()
    
    indices = np.arange(ys.size)
    new_indices = []
    for n,bincount in enumerate(bincounts):
        if bincount == 0:
            continue
        mask = (pots==n)
        idx = indices[mask]
        replicas = np.random.choice(idx, size=max_count-bincount)
        new_indices.append(np.r_[idx, replicas])

    return np.array(new_indices).flatten()


class MemmapDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, filename, shape, dtype, max_size=-1, shuffle=True, z_threshold=None):
        
        if not (os.path.exists(filename) and os.path.isfile(filename)):
            raise ValueError(f'Invalid filename: "{filename}" must exist and be a file')
        
        
        self.dtype = dtype
        
        self.fpr = np.memmap(filename, dtype=dtype, mode='r', shape=tuple(shape))
        
        if z_threshold is None:
            self.indices = np.arange(shape[0])
        else:
            y = self.fpr[:,-1]
            zscores = (y - np.mean(y))/np.std(y)
            self.indices, = np.nonzero(np.abs(zscores) <= z_threshold)
        
        if shuffle:
            np.random.shuffle(self.indices)
        
        self.shape = (self.indices.size, shape[1])
        self.dtype = dtype
        
        self._set_max_size(max_size)
    
    def _set_max_size(self, size):
        if size < 0:
            size = self.shape[0]
        self.__max_size = min(size, self.shape[0])
    
    def _get_max_size(self):
        return self.__max_size
    max_size = property(_get_max_size, _set_max_size)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        xy = np.array(self.fpr[idx,:])
        x = torch.from_numpy(xy[:-1])
        y = torch.from_numpy(xy[ -1].reshape(-1))#.unsqueeze(-1)
        return x, y
    
    def __len__(self):
        return self.max_size
    
    @property
    def targets(self):
        idx = self.indices[:self.max_size]
        return self.fpr[idx,-1]
    
    @property
    def descriptors(self):
        idx = self.indices[:self.max_size]
        return self.fpr[idx,:-1]
    
    def get_indexed_target(self):
        idx = self.indices[:self.max_size]
        target = self.fpr[idx,-1]
        return zip(idx, target)
    
    @staticmethod
    def from_file(datafile, infofile=None, **kwargs):
        import json
        
        if infofile is None:
            root, ext = os.path.splitext(datafile)
            infofile = root + '.json'
        if not (os.path.exists(infofile) and os.path.isfile(infofile)):
            raise ValueError(f'Invalid infofile: "{infofile}" must exist and be a file')
        
        with open(infofile) as fin:
            info_dict = json.load(fin)
        dtype = getattr(np, info_dict['dtype'])
        dataset = MemmapDataset(datafile, info_dict['shape'], dtype, **kwargs)

        return dataset, info_dict
