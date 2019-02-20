import numpy as np
import h5py


class Counter(object):
    def __init__(self):
        self.count = 0
    def __call__(self, name, item):
        if isinstance(item, h5py.Dataset) and name.endswith('/energies'):
            self.count += item.len()

class StdoutHandler(object):
        def __init__(self, *args, **kwargs):
            pass
        def save(self, *args, **kwargs):
            print(args)
            print(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass    

class H5Handler(object):

    def __init__(self, fname, mode='r', compression='gzip', compression_opts=6):
        self.h5 = h5py.File(fname, mode=mode)
        self.compression_opts = compression_opts
        self.compression = compression
        self.filename = fname

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def close(self):
        self.h5.close()
    
    def save(self, group, **data):
        if not isinstance(group, (str, h5py.Group)):
            raise TypeError('group must be an instance of <str> or <h5py.Group>')
        
        if isinstance(group, str):
            group = self.h5.create_group(group)
        
        for key,value in data.items():
            self._save_item(group, key, value)
        
        return group

    def _save_item(self, group, key, value):
        if isinstance(value, list) and value and isinstance(value[0], (str, np.str)):
            value = [v.encode('utf8') for v in value]
        
        elif isinstance(value, str):
            value = [value.encode('utf8')]
            
        group.create_dataset(key, data=value,
            compression=self.compression, compression_opts=self.compression_opts)
    

    def load(self, group, root=None):
        if root is None:
            root = self.h5
        
        if not isinstance(group, (str, h5py.Group)):
            raise TypeError('group must be an instance of <str> or <h5py.Group>')
        
        if isinstance(group, str):
            group = root.get(group)
            
        data = {
            'path': group.name
        }
        for key in group.keys():
            if isinstance(group[key], h5py.Dataset):
                d = self._load_item(group, key)
            else:
                d = self.load(key, group)
            data[key] = d
        return data

    def _load_item(self, group, key):
        # item = group[key].value
        item = group[key][()]
        if isinstance(item, np.ndarray) and isinstance(item[0], np.bytes_):
            item = np.array([b.decode('ascii') for b in item])
        return item

        
    def __iter__(self):
        for group in self.h5.keys():
            data = self.load(group)
            yield data
    
    def iteritems(self):
        f = lambda x: isinstance(x, np.ndarray) and len(x.shape) > 1 
        for ds in self:
            base = {k:v for k,v in ds.items() if not f(v)}
            ndkeys = [k for k,v in ds.items() if f(v)]
            if ndkeys:
                if len(set(len(v) for v in ds.values() if f(v))) > 1:
                    raise Exception('Inconsistent sizes of np.ndarrays')
                for i in range(len(ds[ndkeys[0]])):
                    d = {k:ds[k][i] for k in ndkeys}
                    d.update(base)
                    yield d
            else:
                yield base
    
    # def __len__(self):
    #     return len(self.h5.keys())
    
    def count_groups(self, suffix):
        state = {}
        k = len(suffix)
        def counter(name, item):
            if isinstance(item, h5py.Dataset) and name.endswith(suffix):
                state[name[:-k]] = item.len()
                # print(name, item.len(), n)
        # counter = Counter()
        self.h5.visititems(counter)
        return state
        

if __name__ == "__main__":
    
    # from pyanitools import anidataloader
    import sys

    # dl = anidataloader(sys.argv[1])

    # for group in dl:
    #     print(group['path'], group['species'], group['smiles'])

    with H5Handler(sys.argv[1]) as dl:
        # for k in dl:
        #     print(k)
        #print(len(dl))
        groups = dl.count_groups(sys.argv[2])
        #print(groups)
        print('total number of conformations =', sum(groups.values()))
        print('total number of groups =', len(groups))
        for k,v in groups.items():
            print(k, '=', v)
        
        # for g in dl.iteritems():
        #     print(g)
            # 10
    #dl.close()
