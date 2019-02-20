import mlflow.pytorch
import mlflow
import torch

import argparse
import os

def get_arg_parser(parent=None):
    try:
        parser = parent.add_parser('train',
            help='train network',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    except:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ------------------------------------------------
    group = parser.add_argument_group(title='DATA', description='Source data options')
    group.add_argument(
        '--data-info',
        metavar='FILE',
        type=os.path.abspath,
        help='information on the numpy memmap\'ed file'
    )

    group.add_argument(
        'data',
        type=os.path.abspath,
        help='source numpy memmap\'ed file'
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
    
    # ------------------------------------------------
    group = parser.add_argument_group(title='TRAIN', description='Network training options')
    
    parser.add_argument('--cuda', action='store_true',
                        help='use GPU acceleration')
    group.add_argument('--shuffle', action='store_true',
                        help='shuffle data')
    group.add_argument('--batch-size', type=int, metavar='N',
                        default=128,
                        help='input batch size for training')
    group.add_argument('--test-fraction', type=float, metavar='F',
                        default=0.2, 
                        help='fraction of data to use as test dataset')
    
    group.add_argument('--test-datasets', type=os.path.abspath, metavar='F',
                        nargs='*',
                        default=[],
                        help='test datasets')

    group.add_argument('--epochs', type=int, metavar='N',
                        default=10,
                        help='number of epochs to train')
    group.add_argument('--lr', type=float, metavar='LR',
                        default=0.01,
                        help='learning rate')
    
    group.add_argument('--resume', type=str, metavar='CHK',
                        default=None,
                        help='resume training from checkpoint')
    
    # ------------------------------------------------
    group = parser.add_argument_group(title='OUTPUT', description='Output options')
    group.add_argument('--log-interval', type=int, metavar='N',
                        default=100,
                        help='how many batches to wait before logging training status')
    # group.add_argument('--checkpoint', type=str, metavar='FILE',
    #                     default=None,
    #                     help='reuse model for further training')
    group.add_argument('-o', '--out', type=os.path.abspath, metavar='DIR',
                        default=None, required=True,
                        help='output directory where models will be saved')
    
    group.add_argument('--overwrite', action='store_true',
                        help='force overwriting of files')
    
    # ------------------------------------------------
    group = parser.add_argument_group(title='NETWORK', description='Network options')
    
    group.add_argument('--hidden-layers', nargs='+', metavar='H', type=int,
                        default=[512, 124, 62],
                        help='hidden layers. DIM_IN-[H+]-DIM_OUT')
    #group.add_argument('--mpp-config', metavar='FILE', type=os.path.abspath,
    #                    default=None,
    #                    help='model preprocessor configuration (soap, aev, ...)')
    
    parser.set_defaults(func=main)
    return parser





#------------------------------------------------------------
#------------------------------------------------------------

def main(args):
    
    # --- custom libraries ---    
    from deepshifts.models import get_model, evaluate, save_model
    from deepshifts.utils import  MemmapDataset,Logger,tqdm
    
    # --- third party libraries ---
    from torch.utils.data.dataset import random_split
    from torch.utils.data import DataLoader
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from torchsummary import summary

    # --- builtin libraries ---    
    import functools
    import datetime
    import json
    import os

    #try:
    #    import tplot
    #    l,c = tplot.utils.get_output_size()
    #    plot = tplot.TPlot((28,c))
    #    plot.set_xtick_format('%.2f')
    #    plot.set_ytick_format('%.0f')
    #    plot.set_padding(4)
    #    plot.show_grid()
    #    # tplot.Ansi.enable()
    #except ImportError as ex:
    #    print('Unable to import TPLot')
    #    plot = None
    _plot = None

    # ---------------------------------
    # VALIDATE ARGS
    # ---------------------------------
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    elif os.path.isdir(args.out) and os.listdir(args.out) and not args.overwrite:
        print(f'Error: directory "{args.out}" already exists')
        exit(1)

    if args.resume and not os.path.isfile(args.resume):
        print(f'Error: given checkpoint "{args.resume}" is not a file or does not exist')
        exit(1)
    
    # ---------------------------------
    # GET DATA
    # ---------------------------------
    use_cuda = args.cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset, data_info = MemmapDataset.from_file(
        args.data,
        args.data_info,
        max_size=args.max_size,
        z_threshold=args.z_threshold
    )
    #if args.test_dataset is None:
    #    dataset_size = len(dataset)
    #    test_size = int(dataset_size * args.test_fraction)
    #    train_dataset, test_dataset = random_split(dataset, (dataset_size-test_size, test_size))
    #else:
    #    train_dataset = dataset
    #    test_dataset,_ = MemmapDataset.from_file(
    #        args.test_dataset, None,
    #        max_size=args.max_size,
    #        z_threshold=args.z_threshold
    #    )
    
    dataset_size = len(dataset)
    test_size = int(dataset_size * args.test_fraction)
    train_dataset, test_dataset = random_split(dataset, (dataset_size-test_size, test_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
    test_loader  = DataLoader( test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)


    other_test_loaders = []
    for ds in args.test_datasets:
        name,_ = os.path.splitext(os.path.basename(ds)) 
        other_test_dataset,_ = MemmapDataset.from_file(ds, None)
        other_test_loaders.append((
            name,
            DataLoader(other_test_dataset, batch_size=args.batch_size, **kwargs)
        ))

    #data_loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    #data_loader = test_loader

    # ---------------------------------
    # LOGGING
    # ---------------------------------

    start_time = datetime.datetime.now()
    #run_uuid = f'deepshifts_{start_time.strftime("_%d%b%y_%X")}'
    run_uuid = os.path.basename(args.out)
    
    tensorboard_dir = f'tensorboard/{run_uuid}'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    
    logger = Logger(tensorboard_dir, args.log_interval)
    mlflow.start_run(run_name=run_uuid)

    # output file names
    INFO_FILENAME = os.path.join(args.out, 'info.json')
    MODEL_FILENAME = os.path.join(args.out, 'model.pth')
    STATE_FILENAME = os.path.join(args.out, 'checkpoint.ptar')

    # save args to json file
    with open(INFO_FILENAME, 'wt') as fout:
        data = vars(args)
        data['run_uuid'] = run_uuid
        json.dump(data, fout, indent=4)
    
    # log to mflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)

    print(f'MLFlow identifier = {mlflow.active_run().info.run_uuid}')
    print(f'Tensorboard dir = {logger.log_dir}')
    
    # ---------------------------------
    # MODEL
    # ---------------------------------
    
    NLunit = functools.partial(torch.nn.CELU, alpha=0.1)

    layers = [data_info['shape'][1]-1] + args.hidden_layers + [1]
    model = get_model(*layers, NLunit=NLunit).to(device)
    summary(model, (data_info['shape'][1]-1, ))    


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    loss_fn = torch.nn.SmoothL1Loss(reduction='sum')
    #loss_fn = torch.nn.MSELoss(reduction='sum')

    #full_model = None
    #if args.mpp_config is not None:
    #    with open(args.mpp_config, 'rt') as fin:
    #       mpp_config = json.load(fin)
    #    full_model = get_model(*layers, NLunit=NLunit, mpp=mpp_config)

    # ---------------------------------
    # Checkpoint loading
    # ---------------------------------
    start_epoch = 0
    best_loss = np.inf
    if args.resume:
        print(f'=> loading checkpoint "{args.resume}"')
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_loss = checkpoint['loss']['train']
        print(f'=> resuming from epoch {start_epoch} with loss {best_loss}')

    # ---------------------------------
    # DRIVERS
    # ---------------------------------

    def train(epoch, loader):
        model.train()
        pbar = tqdm(total=len(loader), leave=False, desc=f'epoch {epoch} - training')
        for batch, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if logger.is_log_step(batch):
                avg_loss = loss.data.item()/len(data)
                step = epoch * len(loader) + batch
                logger.log_scalar('batch_train_loss', avg_loss, step)
                logger.log_weights(model, step)
            pbar.update(1)
        pbar.close()
    
    
    def test(epoch, loader, plot=False, name='test'):
        model.eval()
        test_loss = 0.0
        
        if plot:
            nbins = 800
            hist = np.zeros(nbins)
            range_min, range_max = (-20, 20)
            edges = np.linspace(range_min, range_max, nbins+1)

            fig = plt.figure()
            ax = fig.add_subplot(111)

        with torch.no_grad():
            for data, target in tqdm(loader, leave=False, desc=f'epoch {epoch} - testing'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)
                test_loss += loss.item()

                if plot:
                    diff = (output - target).cpu().view(-1)
                    hist += np.histogram(diff, bins=edges)[0]
                    ax.plot(
                        target.cpu().view(-1).numpy(),
                        output.cpu().view(-1).numpy(),
                        'r,'
                    )
        
        if plot:
            hist_sum = hist.sum()
            xp = np.array([0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995])
            p = np.interp(xp, np.cumsum(hist)/hist_sum, edges[1:])
            delta = p[-1] - p[0]
            #ax.set_ylim((p[0]-0.1*delta, p[-1]+0.1*delta))
            ax.set_ylim((-40, 100))
            ax.set_xlim((-40, 100))

            logger.add_figure(f'regression/{name}', fig, epoch)
        
            fig = plt.figure()
            ax = fig.add_subplot(111)

            #hist_sum = hist.sum()
            ax.plot(0.5*(edges[1:] + edges[:-1]), hist)
            #xp = np.array([0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995])
            #p = np.interp(xp, np.cumsum(hist)/hist_sum, edges[1:])
            #delta = p[-1] - p[0]
            ax.set_xlim((p[0]-0.1*delta, p[-1]+0.1*delta))
            logger.add_figure(f'diff/{name}', fig, epoch)
        
        test_loss /= len(loader.dataset)

        return test_loss



    # ---------------------------------
    # Main loop
    # ---------------------------------
    
    
    for epoch in range(start_epoch, args.epochs):

        scheduler.step()
        
        train(epoch, train_loader)
        train_loss = test(epoch, train_loader, plot=True, name='train')
        test_loss  = test(epoch,  test_loader, plot=True, name='test')
        
        lrs = {f'group_{n}':g['lr'] for n,g in enumerate(optimizer.param_groups)}
        logger.add_scalars('loss', {'train':train_loss, 'test': test_loss}, epoch)
        logger.add_scalars('lr', lrs, epoch)
        
        if other_test_loaders:
            others = {
                name:test(epoch,loader,plot=True,name=name)
                    for name,loader in other_test_loaders
            }
            logger.add_scalars('others', others, epoch)
        other_losses = ', '.join(f'{name}: {l:10.6f}' for name,l in others.items())

        print(f'Epoch {epoch+1:3d}/{args.epochs:3d} > train loss: {train_loss:10.6f} | test loss {test_loss:10.6f} [{other_losses}]')
        
        save_model(STATE_FILENAME
            , epoch=epoch
            , model=model
            , layers=layers
            , scheduler=scheduler
            , optimizer=optimizer
            , loss=dict(test=test_loss, train=train_loss)
        )

        if train_loss >= best_loss:
            continue
        
        best_loss = train_loss
        
        # if full_model is not None:
        #     full_model[-1].load_state_dict(model.state_dict())
        #     mlflow.pytorch.log_model(full_model, "models")
            
        #     save_model(MODEL_FILENAME,
        #         mpp_config=mpp_config,
        #         model=full_model,
        #         layers=layers
        #     )

        # save best model
        save_model(MODEL_FILENAME
            , layers=layers
            , model=model
        )
        
        continue
    
        if _plot is not None:
            
            nbins = 800
            hist = np.zeros(nbins)
            range_min, range_max = (-20, 20)
            edges = np.linspace(range_min, range_max, nbins+1)
            plot.clear()

            with torch.no_grad():
                model.eval()
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for data,target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = model(data)
                    diff = (pred - target).cpu().view(-1)
                    hist += np.histogram(diff, bins=edges)[0]
                    ax.plot(target.cpu().view(-1).numpy(), pred.cpu().view(-1).numpy(), 'r,')
                logger.add_figure('regresison', fig, epoch)

                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                hist_sum = hist.sum()
                ax.plot(0.5*(edges[1:] + edges[:-1]), hist)
                xp = np.array([0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995])
                p = np.interp(xp, np.cumsum(hist)/hist_sum, edges[1:])
                delta = p[-1] - p[0]
                ax.set_xlim((p[0]-0.1*delta, p[-1]+0.1*delta)) 

                logger.add_figure('diff', fig, epoch)
                
                # x = 0.5*(edges[1:] + edges[:-1])
                # mask = (hist > 0)
                # ds = plot.bar(x[mask], hist[mask], label=f'Δδ at epoch {epoch+1}')

                # xp = np.array([0.005, 0.05, 0.25, 0.50, 0.75, 0.95, 0.995])
                # p = np.interp(xp, np.cumsum(hist)/hist_sum, edges[1:])
                # ds['percentile'] = [p]
                # delta = p[-1] - p[0]
                
                # plot.set_xlim((p[0]-0.1*delta, p[-1]+0.1*delta)) 
                # print('  Percentiles:')
                # print('\n'.join(('  %8.2f%% -> %9.3f'%t for t in zip(xp*100,p))))
                # print(plot)

                data_size = len(test_loader.dataset)
                n_outliers = data_size - int(hist_sum)
                if n_outliers > 0:
                    print(f'  {n_outliers:_} of {data_size:_} items outside |Δδ|={range_max}')
    

    # close loggers
    logger.close()
    mlflow.end_run()
    
    # output runnig statistics
    end_time = datetime.datetime.now()
    print(f'All done {end_time.strftime("%c")}')
    print(f'  > Ellapsed time = {str(end_time-start_time)}')
