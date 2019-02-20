import ase.io
import ase
import uuid
import io
from rdkit import Chem

HTML_TEMPLATE = """<div>
<canvas id="{canvasid}">Your browser doesn't support HTML5 canvas.</canvas>
<script>
    (function () {{
        var cwc_mol = new ChemDoodle.TransformCanvas3D('{canvasid}', 600, 400);
        cwc_mol.specs.set3DRepresentation('{representation}');
        var molfile = '{mol}';
        var molecule = ChemDoodle.readXYZ(molfile, 1);
        cwc_mol.loadMolecule(molecule);
    }})();
</script>
</div>"""

#cwc_mol.specs.atoms_displayLabels_3D = true


class Molecule(object):
    
    REPR = 'Wireframe'
    
    LINE = 'Line'
    WIREFRAME = 'Wireframe'
    BALL_AND_STICK = 'Ball and Stick'
    VDW_SPHERES = 'van der Waals Spheres'
    STICK = 'Stick'
    
    @staticmethod
    def set_repr3D(repr3d):
        Molecule.REPR = repr3d
    
    #@staticmethod
    #def get_repr3D():
    #    return ('Ball and Stick','van der Waals Spheres',
    #           'Stick', 'Wireframe', 'Line')
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        # for k,v in kwargs.items():
        #     #if isinstance(v, np.ndarray):
        #     #    v = v.tolist()
        #     setattr(self,k,v)
        #if hasattr(self, 'smiles'):
        #    smiles = 
        #    self.smiles = Chem.MolFromSmiles(m.smiles[0])
        
        
    def get_smiles(self):
        if hasattr(self, 'smiles'):
            smiles = Chem.MolToSmiles(self.as_rdkit())
        else:
            smiles = None
        return smiles
    
    def __len__(self):
        return len(self.species)
    
    def as_format(self, format):
        with io.StringIO() as fout:
            mol = ase.Atoms(self.species, self.coords)
            ase.io.write(fout, mol, format=format)
            content = fout.getvalue()
        return content
    
    def __str__(self):
        return self.as_format('xyz')
        #with io.StringIO() as fout:
        #    mol = ase.Atoms(self.species, self.coords)
        #    ase.io.write(fout, mol, format='xyz')
        #    content = fout.getvalue()
        #return content
        
    def _repr_html_(self):
        mol = self.__str__().replace('\n','\\n')
        canvasid = 'canvas_' + hex(uuid.uuid4().fields[-1])
        return HTML_TEMPLATE.format(canvasid=canvasid, mol=mol, representation=self.REPR)
    
    def __repr__(self):
        return self.__str__()
    
    def as_rdkit(self):
        return Chem.MolFromSmiles(self.smiles[0])
    
    @staticmethod
    def from_ase(mol):
        return Molecule(species=mol.get_chemical_symbols(), coords=mol.get_positions())

