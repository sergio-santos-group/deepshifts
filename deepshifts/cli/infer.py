from deepshifts.models import get_model, evaluate
from deepshifts.indexing import Index
import argparse
import torch

def get_arg_parser(root=None):
    try:
        parser = root.add_parser('infer',
            help='use network for inference',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    except:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-m', '--model', metavar='MODEL', type=str,
                        required=True,
                        help='model file')
    parser.add_argument('-i', '--index', metavar='INDEX', type=str,
                        help='index file', default=None)
    parser.add_argument('--plot', action='store_true',
                        help='show plot')
    parser.add_argument('--no-table', action='store_true',
                        help='show plot')
    parser.add_argument('files', nargs='*',
                        help='molecule files in xyz format')
    parser.set_defaults(func=main)
    
    return parser


def get_full_model(model_file):
    import functools
    
    state = torch.load(model_file)
    layers = state['layers']
    config = state['mpp_config']
    model  = state['model']
    
    NLunit = functools.partial(torch.nn.CELU, alpha=0.1)
    
    full_model = get_model(*layers, NLunit=NLunit, mpp=config)
    full_model.load_state_dict(model)
    return full_model


def main(args):
    import ase.io
    from collections import Counter
    import numpy as np

    plot = None
    try:
        import tplot
    except ImportError as ex:
        print('Unable to import TPLot')
    else:
        if args.plot:
            plot = tplot.TPlot()
            plot.set_xtick_format('%d')
            plot.set_ytick_format('%.2f')
            plot.set_padding(2,2,2,0)
            l,c = plot._size
            plot.set_size(24,c)
            plot.show_grid()
            plot.set_tick_position(tplot.Format.LEFT)

            dplot = tplot.TPlot()
            dplot.set_size(16,c)
            dplot.set_padding(2,0,2,2)
            dplot.show_grid()
            dplot.set_ytick_format('%.2f')
            dplot.set_border(tplot.Format.ALL)
            dplot.set_tick_position(tplot.Format.BOTTOM_LEFT)
            

    model = get_full_model(args.model).double()

    # ref = [197.1949,30.8892,32.0125,31.2843,32.0007]
    # x = np.array(x)
    # s = np.array(s)

    def printout(species, shifts, reference, title):
        print(f'# {"id":<3s} {"shifts":>10s} | {"reference":>10s}')
        print('# -------------- | ----------')
        for n,(symbol,shift,ref) in enumerate(zip(species, shifts, reference), start=1):
            if symbol == 'H':
                print(f'  {n:<3d} {shift.item():10.4f}   {ref:10.4f}')
            else:
                print(f'# {n:<3d} {"--":>10s}   {"--":>10s}')
    
    def plotout(sym, calc, ref, title):
        plot.clear().reset()
        # plot.set_ylim((10,40))
        plot.set_title(title)
        m = [n for n,s in enumerate(sym) if s=='H']
        m = np.array(m)
        plot.line(m, calc[m], label='deepshifts', connect=True)
        plot.line(m, ref[m], label='reference', connect=True)
        print(plot, end='')

        dplot.clear().reset()
        dplot.line(m,calc[m]-ref[m], connect=True, label='calc-ref')
        print(dplot)
    
    
    # shifts = evaluate(model, (s,x)).view(-1)
    # printout(s, shifts, ref)

    for file in args.files:
        print(f'#\n# {file}')
        mol = ase.io.read(file)
        sym = mol.get_chemical_symbols()
        xyz = mol.get_positions()
        shifts = evaluate(model, (sym,xyz)).view(-1).numpy()
        ref = mol.get_array('shifts') if mol.has('shifts') else len(shifts)*[-1]
        if not args.no_table:
            printout(sym, shifts, ref, file)

        if plot is not None:
            # plot.clear().reset()
            # m = [n for n,s in enumerate(sym) if s=='H']
            # plot.line(shifts.numpy()[m], label='deepshifts', connect=True)
            # plot.line(ref[m], label='reference', connect=True)
            # print(plot)
            plotout(sym, shifts, ref, file)

    
    if args.index is not None:
        index = Index(args.index)
        
        while True:
            try:
                i = input('index> ')
            except:
                break
            
            try:
                i = int(i)
                data = index[i]
            except Exception as e:
                print(e)
            else:
                sym = data['species']
                xyz = data['coords']
                ref = data['shifts']
                shifts = evaluate(model, (sym,xyz)).view(-1).numpy()
                if not args.no_table:
                    printout(sym, shifts, ref)
                
                if plot is not None:
                    # 8320 = chr(int('2080', 16)+2)

                    # formula = ''.join(s+(str(n) if n>1 else '') for s,n in dict(Counter(sym)).items())
                    formula = ''.join(s+(chr(8320+n) if n>1 else '') for s,n in dict(Counter(sym)).items())

                    plotout(sym, shifts, ref, formula)
                    # plotout(sym, shifts, ref, ''.join(data['smiles']))





    
        

    
