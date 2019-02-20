from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np
import torch



class SOAPComputer(torch.nn.Module):
    def __init__(self, **config):
        super(SOAPComputer, self).__init__()
        self.soap = SOAP(**config)
    
    def __len__(self):
        return self.soap.get_number_of_features()
    
    def forward(self, species_coordinates):
        species, coordinates = species_coordinates
        
        if len(coordinates.shape) == 2:
            coordinates = coordinates[None,:]
        
        mol = Atoms(species, coordinates[0])
        #mask = [n for n,s in enumerate(mol.get_chemical_symbols()) if s==self.target]
        
        descriptors = []
        for coords in coordinates:
            mol.set_positions(coords)
            descriptor = self.soap.create(mol)#, positions=mask)
            descriptors.append(descriptor)
        
        return torch.tensor(descriptors, dtype=torch.float64)



if __name__ == '__main__':
    import ase.io
    import json
    import sys

    with open(sys.argv[1]) as fin:
        config = json.load(fin)
    
    computer = SOAPComputer(**config).double()
     
    Nconfs = 4
    species = list('HHO')
    coords = np.array([[0.0, 0.0, 0.0],[0.8, 0.0, 0.0], [0.1, 0.7, 0.0]])

    print('species, single-conformation')
    print(computer((species, coords)))
    print(computer((species, coords)).shape)

    print('species, multiple-conformations')
    multiple = np.array(Nconfs*[coords])
    print(computer((species, multiple)))
    print(computer((species, multiple)).shape)