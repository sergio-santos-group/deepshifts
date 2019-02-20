from tqdm  import tqdm
import numpy as np
import torch


class Parameter(object):
    def __call__(self):
        raise NotImplementedError()
    

class Choice(Parameter):
    def __init__(self, *options):
        super().__init__()
        self.options = options
    
    def __call__(self):
        x = np.random.choice(self.options)
        #if callable(x):
        if isinstance(x, Parameter):
           x = x()
        return x
    
    def __str__(self):
        return f'Choice({self.options})'

    def __repr__(self):
        return str(self)


class Random(Parameter):
    def __init__(self, distribution, range, step=None):
        super().__init__()
        self.step = step
        self.range = range
        self.distribution = distribution
        self._sampler = getattr(np.random, distribution)
        
        self.__call__()

    def __call__(self):
        x = self._sampler(*self.range)
        if self.step is not None:
            x = round(x//self.step)*self.step
        return x
    
    def __str__(self):
        return f'Random({self.distribution}, range={self.range}, step={self.step})'
    def __repr__(self):
        return str(self)


class Power(Parameter):
    def __init__(self, sampler, base):
        super().__init__()
        self.sampler = sampler
        self.base = base
    
    def __call__(self):
        r = self.sampler()
        return (self.base**r)
    def __str__(self):
        return f'Power({self.sampler}, base={self.base})'
    
    def __repr__(self):
        return str(self)


class HyperBand(object):
    def __init__(self, max_iter=81, eta=3, device=torch.device('cpu'), patience=7):
        
        self.patience = patience
        self.max_iter = max_iter
        self.device = device
        self.eta = eta

        self._hypersamplers = {}
        self._defaults = {}
        self._epoch_based = True
        self._train_dataset = None
        self._validation_dataset = None

    def enable_epoch_based(self):
        self._epoch_based = True

    def enable_batch_based(self):
        self._epoch_based = False

    def set_defaults(self, **kwargs):
        self._defaults.update(kwargs)
    
    def set_train_dataset(self, ds):
        self._train_dataset = ds

    def set_validation_dataset(self, ds):
        self._validation_dataset = ds

    def run(self):
        R = self.max_iter
        eta = self.eta
        s_max = int(np.log(R) / np.log(eta))

        B = (s_max + 1)*R
        best_loss = np.inf
        best = None

        for s in range(s_max,-1,-1):
            r = R * (eta**(-s))
            n = int(np.ceil(int(B/R/(s+1))*(eta**s)))
            T = self.get_hyperparameter_configurations(n)

            tqdm.write('------ s={}, n={} ------'.format(s,n))
            for i in range(s+1):
                ni = int(n * (eta**(-i)))
                ri = int(r * (eta**i))

                # print(i, ni, ri, ni*ri)
                losses = self.run_then_return_val_loss(T, ri)

                k = int(ni/eta)
                if k:
                    top = np.argsort(losses)[:k]
                    #print(top, k, ni, eta)
                    T = T[top]
                    losses = losses[top]
                    tqdm.write(f'[i={i},ni={ni},ri={ri}] => {losses.min()}')
            
            k = np.argmin(losses)
            if losses[k] < best_loss:
                best_loss = losses[k]
                best = T[k]
            
        best['loss'] = best_loss
        return best
    
    def get_hyperparameter_configurations(self, n):
        return np.array([
            self._get_hyperparameter_configuration() for _ in range(n)]
        )


    def run_then_return_val_loss(self, configurations, ri):
        losses = []
        for c in tqdm(configurations, leave=False, desc='loop confs.'):
            losses.append(self._eval(c, ri))
        return np.array(losses)
    
    
    def add(self, target, sampler):
        self._hypersamplers[target] = sampler


    def _get_hyperparameter_configuration(self):
        
        nl = self._sample('nl')
        lr = self._sample('lr')
        layers = self._sample('layer')
        optimizer = self._sample('optimizer')
        evaluator = self._sample('evaluator')
        batch_size = self._sample('batch_size')

        modules = []
        dims = [self._sample('input_dim')] + layers + [self._sample('output_dim')]
        for dim_in, dim_out in zip(dims[:-1],dims[1:]):
            modules.append(torch.nn.Linear(dim_in,dim_out))
            if dim_out > 1:
                modules.append(nl())

        model = torch.nn.Sequential(*modules)
        
        return dict(
            optimizer=optimizer(model.parameters(), lr=lr),
            batch_size=batch_size,
            evaluator=evaluator,
            model=model,
            lr=lr
        )


    def _sample(self, name):
        if name in self._hypersamplers:
            sampler = self._hypersamplers[name]
            if isinstance(sampler, list):
                samples = [f() for f in sampler]
            else:
                samples = sampler()
        else:
            samples = self._defaults[name]
        return samples


    def _eval(self, configuration, max_steps):

        tloader, vloader = self.loader_factory(
            self._train_dataset,
            self._validation_dataset,
            configuration['batch_size']
        )
        
        model = configuration['model'].to(self.device)
        optimizer = configuration['optimizer']
        evaluator = configuration['evaluator']

        counter = 0
        best_loss = np.inf
        nepochs  = max_steps if self._epoch_based else 1
        max_pass = np.inf if self._epoch_based else max_steps
        
        pbar = tqdm(total=nepochs, leave=False, desc='eval  conf.')
        for epoch in range(nepochs):
            self._train_step(tloader, model, evaluator, optimizer, max_pass)
            loss = self._validation_step(vloader, model, evaluator)
            counter += 1
            if loss < best_loss:
                best_loss = loss
                counter = 0
            if counter > self.patience:
                tqdm.write('found stalled optimization')
                break
            pbar.update(1)
        pbar.close()
        
        return best_loss


    def _train_step(self, loader, model, evaluator, optimizer, max_pass=np.inf):
        model.train()

        #pbar = tqdm(total=len(loader), leave=False, desc='train')
        for batch, (data, target) in enumerate(loader):
            if batch > max_pass:
                break
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = evaluator(output, target)
            loss.backward()
            optimizer.step()
            #pbar.update(1)
        #pbar.close()
    

    def _validation_step(self, loader, model, evaluator):
        model.eval()
        validation_loss = 0.0
        #pbar = tqdm(total=len(loader), leave=False, desc='val.')
        with torch.no_grad():
            for batch, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = evaluator(output, target)
                validation_loss += loss.item()
                #pbar.update(1)
        #pbar.close()
        return validation_loss/len(loader.dataset)
    
    def __str__(self):
        def stringify(h,d):
            lines.append('  ' + h + '(')
            for k,v in d.items():
                if isinstance(v, (list,tuple)):
                    v = '[\n        ' + ',\n        '.join(str(x) for x in v) + ']'
                lines.append(f'    {k}: {v}')
            lines.append('  )')

        lines = [self.__class__.__name__ + '(']
        stringify('Hyperparameters', self._hypersamplers)
        stringify('Defaults', self._defaults)
        #lines.append(f'    epoch-based: {self._epoch_based}')
        lines.append(f'  max-resources: {self.max_iter} {"epochs" if self._epoch_based else "batches"}')
        lines.append(f'            eta: {self.eta}')
        lines.append(')')

        return '\n'.join(lines)



if __name__ == '__main__':

    from deepshifts.utils import  MemmapDataset
    from torch.utils.data.dataset import random_split
    from torch.utils.data import DataLoader
    

    def loader_factory(dataset, batch_size):
        return DataLoader(dataset, batch_size, num_workers=1, pin_memory=True)
    

    DATA = '/home/ssantos/deepshifts/data/s1-4.npy'
    BATCH_SIZE = 512
    SIZE = 200_000
    INFO = None

    dataset, data_info = MemmapDataset.from_file(DATA, INFO, max_size=SIZE, z_threshold=3.5)
    
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.3)
    train_dataset, val_dataset = random_split(dataset, (dataset_size-val_size, val_size))
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=1, pin_memory=True)

    def loader_factory(tdataset, vdataset, batch_size):
        return train_loader, val_loader
        #return DataLoader(dataset, batch_size, num_workers=1, pin_memory=True)

    hb = HyperBand(81, 3, device=torch.device('cuda'))
    hb.loader_factory = loader_factory
    hb.enable_epoch_based()
    #hb.enable_batch_based()

    # hb.set_train_dataset(train_dataset)
    # hb.set_validation_dataset(val_dataset)

    hb.set_defaults(
          output_dim=1
        , input_dim=1024
        , batch_size = 512
        , optimizer=torch.optim.Adam
        , evaluator=torch.nn.SmoothL1Loss(reduction='sum')
    )
    
    # learning rate
    hb.add('lr',
        # [0.2, 0.04, 0.008, 0.0016, 0.00032, 6.4e-05]
        Power(Random('uniform', (-1,-6), step=1), 5)
    )

    # non-linear unit
    hb.add('nl',
        Choice(torch.nn.ReLU, torch.nn.CELU)
    )
    
    # # batch_size
    # hb.add('batch_size',
    #     ...
    # )

    # hidden layers
    hb.add('layer', [
        Random('uniform', (500,700), step=10),
        Random('uniform', (400,600), step=10),
        Random('uniform', (100,400), step=10),
        Random('uniform', ( 10,100), step=10),
    ])
    
    print(hb)

    best = hb.run()
    print(best)
    
    # c = Choice(1,2,3,4)
    # print('choice')
    # for n in range(10):
    #     print(n, c())
    
    # c = Random('uniform', (2,8))
    # print(c.distribution)
    # for n in range(10):
    #     print(n, c())
    
    # c = Random('normal', (2,8), step=None)
    # print(c.distribution)
    # for n in range(10):
    #     print('-1',n, c())
    
    # c = Choice(None, Random('uniform', (2,8)))
    # print(c.__class__.__name__)
    # for n in range(10):
    #     print(n, c())
    # print(isinstance(c, Parameter))
