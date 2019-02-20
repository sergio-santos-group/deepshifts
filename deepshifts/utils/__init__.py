from .data import *

from tensorboardX import SummaryWriter
import mlflow
import torch
import glob
import os


__all__ = ['MemmapDataset', 'Logger', 'tqdm', 'file_finder']


class Logger(SummaryWriter):
    def __init__(self, log_dir, log_interval=10, **kwargs):
        super(Logger, self).__init__(log_dir, **kwargs)
        self.log_interval = log_interval
    
    def is_log_step(self, step):
        return self.log_interval and (step % self.log_interval == 0)

    def log_scalar(self, name, value, step):
        self.add_scalar(name, value, step)
        mlflow.log_metric(name, value)

    def log_weights(self, model, step):
        modules = [m for m in model.children() if isinstance(m, torch.nn.Linear)]
        for n,module in enumerate(modules, start=1):
            prefix = f'coefficients/linear{n}-{module.in_features}'
            self.add_histogram(f'{prefix}/weight', module.weight.data, step)
            self.add_histogram(f'{prefix}/bias', module.bias.data, step)


try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    class tqdm(object):
        def __init__(self, *args, **kw):
            pass
        def null(*args, **kw): pass
        def __getattr__(self, _): return self.null
        def write(self, *args, **kw):
            print(*args, **kw)

def file_finder(sources, *suffixes):
    files = []
    for source in sources:
        if os.path.isdir(source):
            for suffix in suffixes:
                files += glob.glob(os.path.join(source, '**/*' + suffix), recursive=True)
        elif os.path.isfile(source) and any([source.endswith(s) for s in suffixes]):
            files.append(source)
        else:
            raise ValueError(f'Invalid file {source}')
    return files
            