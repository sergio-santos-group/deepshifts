from deepshifts.h5handler import H5Handler
from ase import Atoms
import ase.io
import numpy as np
import bisect
import pickle
import os
import io

class Index(object):
    def __init__(self, filename=None):
        self.handlers = {}
        self.idx2file = []
        self.idx2idx = []
        self.idx2ds = []
        self.size = 0

        if filename is not None:
            self.load(filename)
        
    def _add_item(self, container, idx, item):
        if not container:
            idx = -1
        container.append((idx,item))

    def add_file(self, idx, filename):
        self._add_item(self.idx2file, idx, filename)
    
    def add_dataset(self, idx, dataset):
        self._add_item(self.idx2ds, idx, dataset)

    def add_pointer(self, idx, pointer):
        self._add_item(self.idx2idx, idx, pointer)
    
    def set_size(self, size):
        self.size = size
    
    def get_size(self):
        return self.size

    def _get_item(self, container, idx):
        keys = [k for k,_ in container]
        return container[bisect.bisect(keys, idx)-1][1]
    
    def get(self, idx, default=None):
        try:
            item = self.__getitem__(idx)
        except IndexError as e:
            item = default
        return item
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError(f'index must be greater than -1 and not greater than {self.size}')
        
        file = self._get_item(self.idx2file, idx)
        dataset = self._get_item(self.idx2ds, idx)
        pointer = self._get_item(self.idx2idx, idx)

        if file not in self.handlers:
            handler = H5Handler(file)
            self.handlers[file] = handler
        
        item = self.handlers[file].load(dataset)
        for k,v in item.items():
            if isinstance(v, np.ndarray) and len(v.shape) > 1:
                item[k] = v[pointer]
        
        item['dataset'] = dataset
        item['index'] = pointer
        item['file'] = file
        
        return item
    

    def save(self, name):
        with open(name, 'wb') as fout:
            pickle.dump(self.__dict__, fout)
    

    def load(self, name):
        self.close()
        with open(name, 'rb') as fin:
            data = pickle.load(fin)
            self.__dict__.clear()
            self.__dict__.update(data)
    
    def close(self):
        for handler in self.handlers.values():
            handler.close()
        self.handlers = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

    def __iadd__(self, other):
        offset = self.get_size()
        def add(method, container):
            for idx, item in container:
                if idx < 0:
                    idx = 0
                method(idx+offset, item)
        add(self.add_pointer, other.idx2idx)
        add(self.add_dataset, other.idx2ds)
        add(self.add_file, other.idx2file)

        #offset = self.get_size()
        #for idx, item in other.idx2idx:
        #    if idx < 0:
        #        idx = 0
        #    self.add_pointer(idx+offset, item)
        #for idx, item in other.idx2ds:
        #    if idx < 0:
        #        idx = 0
        #    self.add_dataset(idx+offset, item)
        #for idx, item in other.idx2file:
        #    if idx < 0:
        #        idx = 0
        #    self.add_file(idx+offset, item)
        offset += other.get_size()
        self.set_size(offset)
        return self
    
    def __len__(self):
        return self.get_size()

def get_arg_parser(parser=None):
    import argparse

    try:
        parser = parser.add_parser('index',
            help='serve index file through RPC (using xmlrpc)',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    except:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--addr', type=str, default='localhost',
                        help='IP address to use for connection')
    parser.add_argument('--port', type=int, default=8001,
                        help='Port number to use for connection')
    parser.add_argument('index',
                        help='index file to serve')
    parser.set_defaults(func=main)

    return parser


def main(args):
    from xmlrpc.server import SimpleXMLRPCServer
    from xmlrpc.server import SimpleXMLRPCRequestHandler

    class RequestHandler(SimpleXMLRPCRequestHandler):
        rpc_paths = ('/RPC2',)
    
    with SimpleXMLRPCServer((args.addr, args.port),
                    requestHandler=RequestHandler,
                    allow_none=True) as server, \
         Index(args.index) as index:
        server.register_introspection_functions()
        server.register_instance(index)
        print(f'serving {addr} on port {port} ...')
        server.serve_forever()


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    args.func(args)
