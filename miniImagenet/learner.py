from torch import nn
import torch.nn.functional as F 
import torch

class Learner(nn.Module):

    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config 
        self.vars = nn.ParameterList() 
        self.vars_bn = nn.ParameterList()  
        
        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                ## [ch_out, ch_in, kernel_size, kernel_size]
                weight = nn.Parameter(torch.ones(*param[:4])) 
                torch.nn.init.kaiming_normal_(weight)
                self.vars.append(weight) 
                
                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)
                
            elif name == 'linear':
                weight = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(weight)
                self.vars.append(weight)
                bias  = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)
            
            elif name == 'bn':
                weight = nn.Parameter(torch.ones(param[0]))
                self.vars.append(weight)
                bias = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(bias)
                
                ### 
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad = False)
                running_var = nn.Parameter(torch.zeros(param[0]), requires_grad = False)
                
                self.vars_bn.extend([running_mean, running_var])
                
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
                
            else:
                raise NotImplementedError       
    
    
    ## self.net(x_support[i], vars=None, bn_training = True)
    ## x: torch.Size([5, 1, 28, 28])

    def forward(self, x, vars = None, bn_training=True):
        '''
        :param bn_training: set False to not update
        :return: 
        '''
        
        if vars == None:
            vars = self.vars
            
        idx = 0 ; bn_idx = 0
        for name, param in self.config:
            if name == 'conv2d':
                weight, bias = vars[idx], vars[idx + 1]
                x = F.conv2d(x, weight, bias, stride = param[4], padding = param[5]) 
                idx += 2
                
            elif name == 'linear':
                weight, bias = vars[idx], vars[idx + 1]
                x = F.linear(x, weight, bias)
                idx += 2
                
            elif name == 'bn':
                weight, bias = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight= weight, bias = bias, training = bn_training)
                idx += 2
                bn_idx += 2
            
            elif name == 'flatten':
                x = x.view(x.size(0), -1)
            
            elif name == 'relu':
                x = F.relu(x, inplace = [param[0]])
            
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
            
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        
        return x
    
    
    def parameters(self):
        
        return self.vars