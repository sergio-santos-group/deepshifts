import deepshifts.descriptors as descriptors
from deepshifts.utils import tqdm
import torch

    

def get_model(*dims, NLunit=torch.nn.ReLU, mpp=None):
    modules = []
    for dim_in, dim_out in zip(dims[:-1],dims[1:]):
        modules.append(torch.nn.Linear(dim_in, dim_out))
        if dim_out > 1:
            modules.append(NLunit())
    model = torch.nn.Sequential(*modules)
    
    if mpp is not None:
        try:
            desc_class = mpp.get('descriptor_class')
            config = mpp.get('descriptor_config')
            dclass = getattr(descriptors, desc_class)

        except Exception as ex:
            print(ex)
        else:
            model = torch.nn.Sequential(
                dclass(**config),
                model
            )
    
    return model


# def train_step(epoch, model, dloader, optimizer, evaluator, logger, device):
#     '''
#     train the given <model> with data provided by the dataloader <dloader>
#     '''
    
#     state = dict(
#         optimizer=optimizer, epoch=epoch, model=model, data_loader=dloader
#     )


#     model.train()

#     # log current learning rates
#     event_manager.dispatch(ModelEvents.TRAIN_PREEPOCH, **state)
    
#     #for n, group in enumerate(optimizer.param_groups, start=1):
#     #    logger.log_scalar(f'lr_group{n}', group['lr'], epoch)

#     event_manager.dispatch(ModelEvents.TRAIN_EPOCH_START, **state)

#     #pbar = tqdm(total=len(dloader), leave=False, desc=f'epoch {epoch} - training')
#     for batch_idx, (data, target) in enumerate(dloader):
#         event_manager.dispatch(ModelEvents.TRAIN_STEP_START, step=batch_idx, **state)

#         data, target = data.to(device), target.to(device)

#         # perform optimization for current batch
#         optimizer.zero_grad()
#         output = model(data)
#         loss = evaluator(output, target)
#         loss.backward()
#         optimizer.step()
        
#         # # log information
#         # if logger.is_log_step(batch_idx):
#         #     mean_loss = loss.data.item()/len(data)
#         #     step = epoch * len(dloader) + batch_idx
#         #     logger.log_scalar('batch_train_loss', mean_loss, step)
#         #     logger.log_weights(model, step)
#         # pbar.update(1)
        
#         avg_loss = loss.data.item()/len(data)
#         event_manager.dispatch(ModelEvents.TRAIN_STEP_STOP,
#                 step=batch_idx, loss=avg_loss, **state)

#     event_manager.dispatch(ModelEvents.TRAIN_EPOCH_STOP, **state)
#     #pbar.close()


# def test_step(epoch, model, dloader, evaluator, logger, action, device):
#     '''
#     test the given <model> on data provided by the dataloader <dloader>
#     '''
#     model.eval()
#     test_loss = 0.0
    
#     with torch.no_grad(), \
#          tqdm(total=len(dloader), leave=False, desc=f'epoch {epoch} - {action}') as pbar:
#         for data, target in dloader:
#             data, target = data.to(device), target.to(device)
        
#             output = model(data)
#             loss = evaluator(output, target)
#             test_loss += loss.item()
#             pbar.update(1)
    
#     test_loss /= len(dloader.dataset)
#     logger.log_scalar(action+'_loss', test_loss, epoch)

#     return test_loss



def evaluate(model, x):
    model.eval()
    with torch.no_grad():
        y = model(x)
    return y


def save_model(filename, **kwargs):
    state = {}
    for key,value in kwargs.items():
        if hasattr(value, 'state_dict'):
            state[key] = getattr(value, 'state_dict')()
        else:
            state[key] = value
    torch.save(state, filename)


# def load_model(filename):
#     state = torch.load(filename)
#     model = get_model(state['layers'])
#     model.load_state_dict(state['model'])
#     return model, state
