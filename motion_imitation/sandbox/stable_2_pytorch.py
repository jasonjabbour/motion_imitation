import torch as th
import torch.nn as nn
import time

#Define a basic MLP Pytorch Model.
class PyTorchMlp(nn.Module):  

  def __init__(self, n_inputs=4, n_actions=2):
        nn.Module.__init__(self)
        
        # Apply linear transformation y=xA^T + b
        self.fc1 = nn.Linear(n_inputs, 512) #size of input sample. Size of output sample
        self.fc2 = nn.Linear(512, 256)      
        self.fc3 = nn.Linear(256, n_actions)      
        # self.activ_fn = nn.Tanh()
        self.activ_fn = nn.ReLU()
        self.out_activ = nn.Softmax(dim=0)

  def forward(self, x):
        #Should shape x? [120]
        x = self.activ_fn(self.fc1(x)) #8  Relu-> 512
        x = self.activ_fn(self.fc2(x)) #512 Relu-> 256
        x = self.out_activ(self.fc3(x)) #256 Softmax -> 8 
        return x

# Convert weights from 
def copy_mlp_weights(baselines_model):
    torch_mlp = PyTorchMlp(n_inputs=120, n_actions=8)
    model_params = baselines_model.get_parameters()

    policy_keys = [key for key in model_params.keys() if "pi" in key]
    policy_params = [model_params[key] for key in policy_keys]
        
    for (th_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
        # copy parameters from stable baselines model
        param = policy_param.copy()

        # weight of fully connected layer
        if len(param.shape) == 2:
            # transpose parameter
            param = param.T

        #bias
        if 'b' in key:
            # remove all dimensions with size 1
            param = param.squeeze()

        param = th.from_numpy(param)
        pytorch_param.data.copy_(param.data.clone())
        
        
        #param = th.from_numpy(policy_param)    
        #Copies parameters from baselines model to pytorch model
        # print(th_key, key)
        # print(pytorch_param.shape, param.shape, policy_param.shape)
        # pytorch_param.data.copy_(param.data.clone().t())
        
    return torch_mlp


