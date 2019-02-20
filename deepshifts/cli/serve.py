

def get_arg_parser(root=None):
    import argparse
    
    desc = '''
    To load a PDB file:
    PyMOL>r = requests.get('http://localhost:7000/index/210000?fmt=pdb')
    PyMOL>cmd.read_pdbstr(r.json(), objname)
    
    To infer using data from previous request:
    PyMOL>r1 = requests.get('http://localhost:7000/index/210000')
    PyMOL>r2 = requests.post('http://localhost:7000/infer', json=r1.json())
    PyMOL>print r2.json()
    '''
    try:
        parser = root.add_parser('serve',
            help='serve network for inference',
            description=desc,
            formatter_class=argparse.RawDescriptionHelpFormatter)#|ArgumentDefaultsHelpFormatter)
    except:
        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-m', '--model',type=str, required=True,
                        help='model file')
    parser.add_argument('-i', '--index', type=str, default=None,
                        help='index file')
    parser.add_argument('--host', type=str, default='localhost',
                        help='host address')
    parser.add_argument('--port', type=int, default='7000',
                        help='port')
    parser.set_defaults(func=main)

    return parser


def main(args):

    from flask import Flask, request, jsonify, url_for
    from deepshifts.cli.infer import get_full_model
    from deepshifts.models import evaluate
    from deepshifts.indexing import Index
    import numpy as np
    from ase import Atoms
    import ase.io
    import io
    
    # get model
    model = get_full_model(args.model).double()
    
    # get index
    index = Index(args.index)
    
    app = Flask(__name__)
    
    @app.route('/infer', methods=['POST'])
    def infer():
        input_data = request.json
        s = input_data.get('species', None)
        # x = input_data.get('coordinates', None)
        x = input_data.get('coords', None)
        if s is None or x is None:
            return 'must specify coordinates (shape Nx3) and species (N)', 422 
        shifts = evaluate(model, (s,x))
        shifts = shifts.view(-1).numpy().tolist()
        #input_data.update(shifts=shifts)

        return jsonify(shifts)
    

    @app.route('/index/<int:item>', methods=['GET'])
    def get(item):
        
        try:
            data = index[item]
        except IndexError as ex:
            return str(ex), 422
        
        print(data)
        fmt = request.args.get('fmt', None)
        if fmt is not None:
            try:
                fout = io.StringIO()
                mol = Atoms(data['species'], data['coords'])
                mol.set_array('shifts', data['shifts'])
                ase.io.write(fout, mol, format=fmt)
                data['request'] = fout.getvalue()
                data['format'] = fmt
            except Exception as ex:
                return str(ex), 422
            finally:
                fout.close()
        
        if isinstance(data, dict):
            for k,v in data.items():
                if hasattr(v, 'tolist'):
                    data[k] = v.tolist()
        
        return jsonify(data)


    @app.route('/')
    def home():
        return f'inference/get endpoints available at {url_for("infer")}/{url_for("get")}'



    app.run(host=args.host, port=args.port, debug=True, extra_files=args.model)



