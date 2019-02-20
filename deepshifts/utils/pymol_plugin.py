from pymol import cmd
import requests
import pprint

@cmd.extend
def deepshifts_load(index, objname=None, host='localhost', port=7000, atom_props=None):
    '''
    deepshifts_load(index, objname=None, host='localhost', port=7000)
    '''

    # get data
    req = requests.get('http://{}:{}/index/{}?fmt=proteindatabank'.format(
        host, port, index
    ))
    data = req.json()

    # load object
    if objname is None:
        objname = 'obj-' + str(index)
    cmd.read_pdbstr(data['request'], objname)
    cmd.set_title(objname, 1, data['energy'][0])

    # load properties
    if atom_props is None:
        atom_props = ''

    props = {p:data[p] for p in atom_props.split() if p in data}

    for prop,values in props.iteritems():
        #for n,value in enumerate(values):
        #    selection = 'id %d in %s'%(n, objname)
        #    cmd.set_atom_property(prop, value, selection)
        cmd.iterate(objname, 'p["{}"]=l.pop(0)'.format(prop), space={'l': values})
    pprint.pprint(data)


@cmd.extend
def deepshifts_infer(selection, host='localhost', port=7000):
    model = cmd.get_model(selection, cmd.get_state())

    data = {
        'species': [atom.symbol for atom in model.atom],
        'coords' : [atom.coord for atom in model.atom]
    }
    
    req = requests.post('http://{}:{}/infer'.format(host,port), json=data)
    shifts = req.json()
    
    for n,(s,y) in enumerate(zip(data['species'],shifts)):
        print(n,s,y)
    cmd.iterate(selection, 'p["deepshifts"]=l.pop(0)', space={'l': shifts})

        
# @cmd.extend
# def deepshifts_connect(addr='localhost', port=8001):
#     uri = 'http://{}:{}'.format(addr,port)
#     # if cmd.rpc_client is not None:
#     #     print 'Discarding previous connection'
#     #     cmd.rpc_client.close()
#     cmd.rpc_client = xmlrpclib.ServerProxy(uri)


# @cmd.extend
# def deepshifts_load(map_file):
#     if cmd.rpc_client is None:
#         print 'No connection is available'
#         return
#     # else:
#     #     print 'Discarding previous connection'
#     #     cmd.rpc_client.close()
#     cmd.rpc_client.load(map_file)
#     print 'map file = ', map_file
#     print 'map size = ', cmd.rpc_client.get_size()


# @cmd.extend
# def deepshifts_get(index, objname=None):
#     if cmd.rpc_client is None:
#         print 'No connection is available'
#         return
#     index = int(index)
#     obj_str = cmd.rpc_client.get_pdb(index)
#     print obj_str
#     if objname is None:
#         objname = 'dsobj{}'.format(index)
#     cmd.read_pdbstr(obj_str, objname)
