import argparse
import torch
import os

def get_arg_parser(parent=None):
    try:
        parser = parent.add_parser('hyperband',
            help='optimize network using HyperBand',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    except:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cuda', action='store_true',
                        help='use GPU acceleration')
    
    # ------------------------------------------------
    group = parser.add_argument_group(title='DATA', description='Source data options')

    group.add_argument(
        'data',
        type=os.path.abspath,
        help='source numpy memmap\'ed file'
    )

    group.add_argument(
        '--data-info',
        metavar='FILE',
        type=os.path.abspath,
        help='information on the numpy memmap\'ed file'
    )
    group.add_argument(
        '--max-size',
        default=-1, 
        type=int, metavar='N',
        help='max number of data points to use'
    )
    
    group.add_argument(
        '--z-threshold',
        type=float,
        metavar='Z',
        default=None, 
        help='z-score threshold for outlier detection'
    )

    group.add_argument(
        '--test-fraction',
        type=float,
        metavar='F',
        default=0.3,
        help='fraction of data to use as test dataset'
    )
    
    
    
    # ------------------------------------------------
    group = parser.add_argument_group(
        title='TRAIN',
        description='Network training options. Non-provided ' + \
                    'option will be optimized.'
        )
    
    group.add_argument(
        '--batch-size',
        type=int,
        metavar='N',
        default=None,
        help='Batch size for training'
    )

    # group.add_argument('--epochs', type=int, metavar='N',
    #                     default=10,
    #                     help='number of epochs to train')
    group.add_argument('--lr', type=float, metavar='LR',
                        default=None,
                        help='learning rate')
    group.add_argument('--optimizer', type=str,
                        #default=None,
                        choices=['Adam', 'SGD'],
                        help='network optimization engine')
    
    # group.add_argument('--resume', type=str, metavar='CHK',
    #                     default=None,
    #                     help='resume training from checkpoint')
    
    # ------------------------------------------------
    group = parser.add_argument_group(title='OUTPUT', description='Output options')
    # group.add_argument('--log-interval', type=int, metavar='N',
    #                     default=100,
    #                     help='how many batches to wait before logging training status')
    # # group.add_argument('--checkpoint', type=str, metavar='FILE',
    # #                     default=None,
    # #                     help='reuse model for further training')
    group.add_argument('-o', '--out', type=os.path.abspath, metavar='FILE',
                        default=None, required=True,
                        help='name of the file onto which the best model will be saved')
    
    
    # ------------------------------------------------
    group = parser.add_argument_group(
        title='HYPERBAND', description='HyperBand options')
    
    group.add_argument('--batch-based', action='store_true',
                        help='batch-based resource allocation')

    group.add_argument('--resources', type=int, default=81, metavar='R',
                        help='maximum available resources')

    group.add_argument('--eta', type=int, default=3, metavar='η',
                        help='proportion of configurations discarded in each round')

    # group.add_argument('--hidden-layers', nargs='+', metavar='H', type=int,
    #                     default=[512, 124, 62],
    #                     help='hidden layers. DIM_IN-[H+]-DIM_OUT')
    # #group.add_argument('--mpp-config', metavar='FILE', type=os.path.abspath,
    # #                    default=None,
    # #                    help='model preprocessor configuration (soap, aev, ...)')
    
    parser.set_defaults(func=main)
    return parser





#------------------------------------------------------------
#------------------------------------------------------------

def main(args):
    
    from torch.utils.data.dataset import random_split
    from deepshifts.utils import  MemmapDataset
    from deepshifts.models import save_model    
    from torch.utils.data import DataLoader
    from deepshifts import hyperband
    import datetime
    import torch

    start_time = datetime.datetime.now()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # =================================
    # LOAD DATA
    # =================================
    dataset, data_info = MemmapDataset.from_file(
        args.data, args.data_info,
        max_size=args.max_size, z_threshold=args.z_threshold
    )

    dataset_size = len(dataset)
    val_size = int(dataset_size * args.test_fraction)
    train_dataset, val_dataset = random_split(dataset, (dataset_size-val_size, val_size))

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.batch_size is  None:
        def loader_factory(tdataset, vdataset, batch_size):
            train_loader = DataLoader(train_dataset, batch_size, **loader_kwargs)
            val_loader = DataLoader(val_dataset, batch_size, **loader_kwargs)
            return train_loader, val_loader
    else:
        train_loader = DataLoader(train_dataset, args.batch_size, **loader_kwargs)
        val_loader = DataLoader(val_dataset, args.batch_size, **loader_kwargs)

        def loader_factory(tdataset, vdataset, batch_size):
            return train_loader, val_loader

    # =================================

    hb = hyperband.HyperBand(
        args.resources, args.eta,
        device=device
    )

    hb.set_defaults(
        optimizer=getattr(torch.optim, str(args.optimizer), None)
        , evaluator=torch.nn.SmoothL1Loss(reduction='sum')
        , input_dim=data_info['shape'][1]-1
        , batch_size = args.batch_size
        , output_dim=1
        , lr = args.lr
    )

    hb.loader_factory = loader_factory

    # learning rate
    if args.lr is None:
        hb.add('lr',
            # [0.2, 0.04, 0.008, 0.0016, 0.00032, 6.4e-05]
            hyperband.Power(hyperband.Random('uniform', (-1,-6), step=1), 5)
        )

    # batch size
    if args.batch_size is None:
        hb.add('batch_size',
            hyperband.Random('uniform', (100, 1000), step=100)
        )
    
    if args.optimizer is None:
        hb.add('optimizer',
            hyperband.Choice(torch.optim.Adam, torch.optim.SGD)
        )

    # non-linear unit
    hb.add('nl',
        hyperband.Choice(torch.nn.ReLU, torch.nn.CELU)
    )

    # hidden layers
    hb.add('layer', [
        hyperband.Random('uniform', (500,700), step=10),
        hyperband.Random('uniform', (400,600), step=10),
        hyperband.Random('uniform', (100,400), step=10),
        hyperband.Random('uniform', ( 10,100), step=10),
    ])

    if args.batch_based:
        hb.enable_batch_based()
    else:
        hb.enable_epoch_based()

    print(hb)
    best = hb.run()
    print(best)

    root, ext = os.path.splitext(args.out)
    with open(root+'.log', 'wt') as fout:
        fout.write(str(hb))
        fout.write('\n')
        fout.write(str(best))

    save_model(args.out, **best)

    # =================================
    end_time = datetime.datetime.now()
    print(f'All done {end_time.strftime("%c")}')
    print(f'  > Ellapsed time = {str(end_time-start_time)}')

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    exit(args.func(args))
