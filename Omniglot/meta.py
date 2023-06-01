from copy import deepcopy
from torch import nn
from learner import Learner
import torch
import torch.nn.functional as F 
import numpy as np

printed = True

def Softmax_temp(input, t=1):
    
    global printed
    
    if printed:
        print('\n',"Temp : ",t)
        printed = False
        
    input = input.T
    ex = torch.exp(input/t)
    sum = torch.sum(ex, axis=0)
    output = ex / sum
    return output.T

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = Softmax_temp(X)
    log_likelihood = -torch.log(p[range(m),y])
    loss = torch.sum(log_likelihood) / m
    return loss

class Meta(nn.Module):
    """
    Meta-Learner
    """
    def __init__(self, config):
        super(Meta, self).__init__()   
        self.update_lr = 0.1 ## learner中的学习率，即\alpha
        self.meta_lr = 1e-3 ## meta-learner的学习率，即\beta
        self.n_way = 5 ## 5种类型
        self.k_shot = 5 ## 一个样本
        self.k_query = 15 ## 15个查询样本
        self.task_num = 4 ## 每轮抽8个任务进行训练
        self.update_step = 5 ## task-level inner update steps
        self.update_step_test = 5 ## 用在finetunning这个函数中
        
        self.net = Learner(config) ## base-learner
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr = self.meta_lr)
        
        
    def forward(self, x_support, y_support, x_query, y_query):
        """
        :param x_spt:   torch.Size([8, 5, 1, 28, 28])
        :param y_spt:   torch.Size([8, 5])
        :param x_qry:   torch.Size([8, 75, 1, 28, 28])
        :param y_qry:   torch.Size([8, 75])
        :return:
        N-way-K-shot
        """
        task_num, ways, shots, h, w = x_support.size()
#         print("Meta forward")
        querysz = x_query.size(1)## 75 = 15*5
        losses_q = [0 for _ in range(self.update_step +1)] ## losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step +1)]
        
        for i in range(task_num):    
            
            ## 第0步更新
            logits = self.net(x_support[i], vars=None, bn_training = True)## return 一个经过各层计算后的y
            # logits = Softmax_temp(torch.tensor(logits))
            # print(logits)
            ## logits : 5*5的tensor
            loss = cross_entropy(logits, y_support[i])
            #loss = Variable(loss, requires_grad = True)
            #print(loss)
            #loss = F.cross_entropy(logits, y_support[i])  ## 计算Loss值
            grad = torch.autograd.grad(loss, self.net.parameters()) ##计算梯度。如果输入x，输出是y，则求y关于x的导数（梯度）
            tuples = zip(grad, self.net.parameters() ) ##将梯度grad和参数\theta一一对应起来
            ## fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
            fast_weights = list( map(lambda p: p[1] - self.update_lr * p[0], tuples) )
            
            ### Test on the query set and calculate the accuracy
            ## This step uses the parameters before the update
            with torch.no_grad():
                logits_q = self.net(x_query[i], self.net.parameters(), bn_training = True) ## logits_q :torch.Size([75, 5])
                #print(x_query[i].shape)
                loss_q = F.cross_entropy(logits_q, y_query[i]) ## y_query : torch.Size([75])
                #loss_q = cross_entropy(logits_q, y_query[i])
                losses_q[0] += loss_q ##将loss存在数组的第一个位置
                pred_q = F.softmax(logits_q, dim = 1).argmax(dim=1) ## size = (75)
                correct = torch.eq(pred_q, y_query[i]).sum().item()## item()取出tensor中的数字
                corrects[0] += correct
            
            ### Test on the query set and calculate the accuracy
            ## This step uses the updated parameters
            with torch.no_grad():
                logits_q = self.net(x_query[i], fast_weights, bn_training = True)
                loss_q = F.cross_entropy(logits_q, y_query[i])
                #loss_q = cross_entropy(logits_q, y_query[i])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim = 1).argmax(dim=1)
                correct = torch.eq(pred_q, y_query[i]).sum().item()
                corrects[1] += correct
             
            
            for k in range(1, self.update_step):
                logits = self.net(x_support[i], fast_weights, bn_training =True)
                loss = cross_entropy(logits, y_support[i])
                #loss = F.cross_entropy(logits, y_support[i])
                grad = torch.autograd.grad(loss, fast_weights)
                tuples = zip(grad,fast_weights)
                fast_weights = list(map(lambda p:p[1] - self.update_lr * p[0], tuples))
                
                if k < self.update_step - 1:
                    with torch.no_grad():   
                        logits_q = self.net(x_query[i], fast_weights, bn_training = True)
                        #loss_q = cross_entropy(logits_q, y_query[i])
                        loss_q = F.cross_entropy(logits_q, y_query[i])
                        losses_q[k+1] += loss_q
                        
                else:
                    logits_q = self.net(x_query[i], fast_weights, bn_training = True)
                    #loss_q = cross_entropy(logits_q, y_query[i])
                    loss_q = F.cross_entropy(logits_q, y_query[i])
                    losses_q[k+1] += loss_q
                
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim = 1)
                    correct = torch.eq(pred_q, y_query[i]).sum().item()
                    corrects[k+1] += correct
                    
        ## After a group of tasks ends, find an average loss
        loss_q = losses_q[-1] / task_num
        self.meta_optim.zero_grad() ## 梯度清零
        loss_q.backward() ## 计算梯度
        self.meta_optim.step()  # Use the set optimization method to iterate the model parameters, 
                                # this step is the meta step iteration
        
        accs = np.array(corrects) / (querysz * task_num) 
        
        return accs
        
    
    def finetunning(self, x_support, y_support, x_query, y_query):
        assert len(x_support.shape) == 4
        
        querysz = x_query.size(0)
        
        corrects = [0 for _ in range(self.update_step_test + 1)]
        
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        
        logits = net(x_support)
        loss = cross_entropy(logits, y_support)
        #loss = F.cross_entropy(logits, y_support)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))      
        
        ## Accuracy before starting training
        with torch.no_grad():
            logits_q = net(x_query, net.parameters(), bn_training = True)
            pred_q = F.softmax(logits_q, dim =1).argmax(dim=1)
            correct = torch.eq(pred_q, y_query).sum().item()
            corrects[0] += correct
         
        ## Accuracy after training
        with torch.no_grad():
            logits_q = net(x_query, fast_weights, bn_training = True)
            pred_q = F.softmax(logits_q, dim = 1).argmax(dim=1)
            correct = torch.eq(pred_q, y_query).sum().item()
            corrects[1] += correct
            
        for k in range(1, self.update_step_test):
            logits = net(x_support, fast_weights, bn_training=True)
            loss = cross_entropy(logits, y_support)
            #loss = F.cross_entropy(logits, y_support)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            
            logits_q = net(x_query, fast_weights, bn_training=True)
            #loss_q = cross_entropy(logits_q, y_query)
            loss_q = F.cross_entropy(logits_q, y_query)
            
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim =1).argmax(dim=1)
                correct = torch.eq(pred_q, y_query).sum().item()
                corrects[k+1] += correct
                
        del net
        
        accs = np.array(corrects) / querysz
        
        return accs