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
