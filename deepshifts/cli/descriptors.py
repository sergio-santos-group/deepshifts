
def get_arg_parser(root=None):
    from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter

    try:
        parser = root.add_parser('descriptors',
            help='descriptor calculator',
            description='request calculation of atomic descriptors',
            formatter_class=ArgumentDefaultsHelpFormatter)
    except:
        parser = ArgumentParser(
            description='request calculation of atomic descriptors',
            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('files', type=str, nargs='+',
        help=(f'H5DF source files. '
              f'If a directory is provided, all files ending in *.{{h5,h5df}} will '
              f'be recursively found and processed'))

    parser.add_argument('--start', type=int,
                        default=0,
                        help='slice start')
    parser.add_argument('--stop', type=int,
                        default=None,
                        help='slice stop')
    parser.add_argument('--step', type=int,
                        default=1,
                        help='slice step')
    parser.add_argument('--sort', action='store_true',
                        default=False,
                        help='sort items in each group according to their energy')
    parser.add_argument('--out', type=str, action='store', metavar='FILE',
                        default='memmap.npy',
                        help='output filename')
    #parser.add_argument('--type', type=str.upper, required=True,
    #                    choices=['AEV', 'SOAP'],
    #                    help='descriptor type')
    parser.add_argument('--config', type=str, required=True,
                        help='descriptor configuration file')
    parser.add_argument('--double', action='store_true',
                        help='request output as type <double>')
    parser.add_argument('--probe', type=str, default='H',
                        help='probe symbols')
    
    parser.set_defaults(func=main)

    return parser


def main(args):
    from deepshifts.utils import tqdm, file_finder
    import deepshifts.descriptors as descriptors
    from deepshifts.h5handler import H5Handler
    from deepshifts.indexing import Index
    import numpy as np
    import datetime
    import json
    import sys
    import os
    
    start_time = datetime.datetime.now()
    
    # ---------------------------------
    # Get all unique files
    # ---------------------------------
    #files = []
    #for file in args.files:
    #   if os.path.isdir(file):
    #       files += glob.glob(os.path.join(file, '**/*.h5'), recursive=True)
    #       files += glob.glob(os.path.join(file, '**/*.h5df'), recursive=True)
    #   elif os.path.isfile(file) and (file.endswith('h5') or file.endswith('h5df')):
    #       files.append(file)
    #   else:
    #       raise ValueError(f'Invalid file {file}')
    files = file_finder(args.files, '.h5', '.h5df')
    files = sorted(set(files))

    # ---------------------------------
    # Get descriptor computer
    # ---------------------------------
    with open(args.config, 'rt') as fin:
        desc_config = json.load(fin)
    
    #klass = SOAPComputer if args.type == 'SOAP' else AEVComputer
    #desc_computer = klass(**desc_config).double()
    #klass = getattr(descriptors, '{}Computer'.format(args.type))
    desc_type = desc_config['type'].upper()
    klass = getattr(descriptors, '{}Computer'.format(desc_type))
    desc_computer = klass(**desc_config).double()

    #if args.double:
    #    desc_computer = desc_computer.double()
    
    # ---------------------------------
    # MAIN WORKHORSE
    # ---------------------------------

    # 1st PASS
    #   do a first pass through all files to
    #   determine full AEV matrix size
    h5handlers = [H5Handler(file) for file in files]
    slicer = slice(args.start, args.stop, args.step)
    size = 0
    probe = args.probe
    with tqdm(desc='determining full matrix size', leave=False) as pbar:
        for h5handler in h5handlers:
            for group in h5handler:
                # number of 'H' in this group (in each structure)
                nhs = sum(group['species']==probe)
                nstructs = len(group['coords'])
                size += nhs * len(range(*slicer.indices(nstructs)))
                pbar.update(1)

    dtype = np.float_ if args.double else np.float32
    shape = (size, len(desc_computer)+1)
    tqdm.write(f'full descriptor matrix: shape={shape}, type={dtype.__name__}', file=sys.stderr)
    
    # 2nd PASS
    #   do a second pass to calculate
    #   full AEV matrix and dump it to file
    fp = np.memmap(args.out, dtype=dtype, mode='w+', shape=shape)
    index = Index()

    idx = 0
    with tqdm(total=size, leave=False) as pbar:
        for h5handler in h5handlers:
            index.add_file(idx, os.path.abspath(h5handler.filename))
            pbar.set_description(h5handler.filename)
            for dataset in h5handler:
                index.add_dataset(idx, dataset['path'])

                nstructs = len(dataset['coords'])
                # take only required structures
                #  so far, shapes are:
                #     coords.shape = (Nstructs, Natoms, 3) 
                #     shifts.shape = (Nstructs, Natoms)
                if args.sort:
                    indices = np.argsort(dataset['energy'].flatten())
                    indices = indices[slicer]
                else:
                    indices = list(range(*slicer.indices(nstructs)))
                nstructs = len(indices)
                
                #  after slicing, we have:
                #         nstructs = slice_size
                #     coords.shape = (nstructs, Natoms, 3) 
                #     shifts.shape = (nstructs, Natoms)
                coords = dataset['coords'][indices]
                shifts = dataset['shifts'][indices]
                #anisos = dataset['anisotropic'][indices]
                
                
                # take species and identify hydrogens
                #   species.shape = (Natoms,)
                species = dataset['species']
                mask, = np.nonzero(species == probe)
                nhs = len(mask)

                # compute descriptors and save to file
                descriptors = desc_computer((species, coords))
                
                for i in range(nstructs):
                    index.add_pointer(idx, indices[i])
                    fp[idx:idx+nhs,:] = np.c_[descriptors[i,mask], shifts[i,mask]].astype(dtype)
                    pbar.update(nhs)
                    idx += nhs

            # close this handler since it will no longer be needed
            h5handler.close()
        
    # flush data to file
    del fp

    # ---------------------------------
    # FINAL LOG
    # ---------------------------------
    
    # set index full size and save to file
    root, ext = os.path.splitext(args.out)
    index.set_size(int(idx))
    index.save(root + '.map')

    # write a log file containing the information on the conversion details
    with open(root + '.json', 'wt') as fout:
        data = dict(
            files=files
            , shape=list(map(int,shape))
            , dtype=dtype.__name__
            , slice=dict(
                start=args.start
                , stop=args.stop
                , step=args.step
            )
            , sort=args.sort
            , descriptor_class=desc_computer.__class__.__name__
            , descriptor_config=desc_config
        )
        #data.update(desc_config) 
        json.dump(data, fout, indent=4)
    
    end_time = datetime.datetime.now()
    tqdm.write(f'All done {end_time.strftime("%c")}', file=sys.stderr)
    tqdm.write(f'  > Ellapsed time = {str(end_time-start_time)}')
    



if __name__ == "__main__":
    
    parser = get_arg_parser()
    args = parser.parse_args()

    args.func(args)
