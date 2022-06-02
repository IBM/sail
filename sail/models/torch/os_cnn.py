"""
Omni-Scale 1D-CNN

Code adapted from https://github.com/Wensi-Tang/OS-CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn 
import torch.optim as optim
from skorch.classifier import NeuralNetClassifier


def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1): 
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer/(in_channel*sum(prime_list)))
    return out_channel_expect


def generate_layer_parameter_list(start,end,paramenter_number_of_layer_list, in_channel = 1):
    prime_list = get_Prime_number_in_a_range(start, end)
    # if prime_list == []:
    #     # print('start = ',start, 'which is larger than end = ', end)
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        
        tuples_in_layer= []
        for prime in prime_list:
            tuples_in_layer.append((in_channel,out_channel,prime))
        in_channel =  len(prime_list)*out_channel
        
        layer_parameter_list.append(tuples_in_layer)
    
    tuples_in_layer_last = []
    first_out_channel = len(prime_list)*get_out_channel_number(paramenter_number_of_layer_list[0], input_in_channel, prime_list)
    tuples_in_layer_last.append((in_channel,first_out_channel,start))
    tuples_in_layer_last.append((in_channel,first_out_channel,start+1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list


def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now


def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)

    
class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        result = F.relu(result_3)
        return result    
    
    
class _OS_CNN(nn.Module):
    def __init__(
        self,
        n_class, #number of classes to predict
        input_channel,
        receptive_field_shape,
        start_kernel_size = 1,
        max_kernel_size = 89,
        paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128],
        few_shot = False
    ):
        super(_OS_CNN, self).__init__()
        self.few_shot = few_shot
        self.start_kernel_size = start_kernel_size
        self.max_kernel_size = max_kernel_size
        self.paramenter_number_of_layer_list = paramenter_number_of_layer_list
        self.input_channel = input_channel
        self.receptive_field_shape = receptive_field_shape
        
        layer_parameter_list = generate_layer_parameter_list(self.start_kernel_size,
                                                     receptive_field_shape,
                                                     self.paramenter_number_of_layer_list,
                                                     in_channel = self.input_channel)
        self.layer_list = []
        
        # building each model layer from a list of parameter
        for i in range(len(layer_parameter_list)):
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            
            # print([layer.shape for layer in layer.parameters()])
            self.layer_list.append(layer)
        
        self.net = nn.Sequential(*self.layer_list)
            
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        
        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1] 
            
        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):
        
        X = self.net(X)

        X = self.averagepool(X)
        X = X.squeeze_(-1)

        if not self.few_shot:
            X = self.hidden(X)
        return X
    

class OS_CNN_CLassifier(NeuralNetClassifier):
    """
    Basic OS_CNN model
    
    Args:
        n_class: Number of classes to predict]
        input_channel: Number of input channels. varies depending on data type (univariate and multivariate)
        receptive_field_shape: Scales of receptive fields to avoid scale tuning 
        start_kernel_size: Start kernel size. Must be a prime number 
        max_kernel_size: Maximum kernel size. Must be a prime number
        paramenter_number_of_layer_list: OS block layer size
        device: cpu or gpu
        max_epochs: Epoch number
        batch_size: Size of the batch
        criterion: Loss function
        optimizer: Optimizer
    
    """
    def __init__(
        self,
        n_class, 
        input_channel,
        receptive_field_shape,
        start_kernel_size = 1,
        max_kernel_size = 89,
        paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128],
        device = "cpu",
        few_shot = False,
        max_epochs = 20,
        **kwargs):
        super(OS_CNN_CLassifier, self).__init__(
            module = _OS_CNN,
            module__n_class = n_class,
            module__input_channel = input_channel,
            module__receptive_field_shape = receptive_field_shape,
            module__start_kernel_size = start_kernel_size,
            module__max_kernel_size = max_kernel_size,
            module__paramenter_number_of_layer_list = paramenter_number_of_layer_list,
            module__few_shot = few_shot,
            device = device,
            max_epochs=max_epochs, 
            batch_size=16,
            criterion = nn.CrossEntropyLoss,
            optimizer = optim.Adam,
            **kwargs
        )
        