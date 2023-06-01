from copy import deepcopy
from torch import nn
from learner import Learner
import torch
import torch.nn.functional as F 
import numpy as np

class Meta(nn.Module):
    """
    Meta-Learner
    """
    def __init__(self, config, temp):
        super(Meta, self).__init__()   
        self.update_lr = 0.1 
        self.meta_lr = 1e-3 
        self.n_way = 5 
        self.k_shot = 5 
        self.k_query = 15 
        self.task_num = 4 
        self.update_step = 5
        self.update_step_test = 5 
        self.temp = temp
        
        self.net = Learner(config) ## base-learner
        self.meta_optim = torch.optim.Adam(self.net.parameters(), lr = self.meta_lr)        
        self.printed = True

    def cross_entropy(self,X,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector. 
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        
        def Softmax_temp(self,input_data):
            
            if self.printed == True:
                print('\n',"Temperature parameter : ",self.temp)
                self.printed = False
                
            input_data = input_data.T
            ex = torch.exp(input_data/self.temp)
            sum = torch.sum(ex, axis=0)
            output = ex / sum
            return output.T
        
        m = y.shape[0]
        p = Softmax_temp(self,X)
        log_likelihood = -torch.log(p[range(m),y])
        loss = torch.sum(log_likelihood) / m
        return loss
        
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
        # print("Meta forward")
        querysz = x_query.size(1)## 75 = 15*5
        losses_q = [0 for _ in range(self.update_step +1)] ## losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step +1)]
        
        for i in range(task_num):    
            
            logits = self.net(x_support[i], vars=None, bn_training = True)
            # logits = Softmax_temp(torch.tensor(logits))
            # print(logits)
            ## logits : 5*5 tensor
            loss = self.cross_entropy(logits, y_support[i])
            #loss = Variable(loss, requires_grad = True)
            #print(loss)
            #loss = F.cross_entropy(logits, y_support[i])  
            grad = torch.autograd.grad(loss, self.net.parameters()) 
            tuples = zip(grad, self.net.parameters() ) 
            ## fast_weights
            fast_weights = list( map(lambda p: p[1] - self.update_lr * p[0], tuples) )
            
            ### Test on the query set and calculate the accuracy
            ## This step uses the parameters before the update
            with torch.no_grad():
                logits_q = self.net(x_query[i], self.net.parameters(), bn_training = True) ## logits_q :torch.Size([75, 5])
                #print(x_query[i].shape)
                loss_q = F.cross_entropy(logits_q, y_query[i]) ## y_query : torch.Size([75])
                #loss_q = cross_entropy(logits_q, y_query[i])
                losses_q[0] += loss_q 
                pred_q = F.softmax(logits_q, dim = 1).argmax(dim=1) 
                correct = torch.eq(pred_q, y_query[i]).sum().item()
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
                loss = self.cross_entropy(logits, y_support[i])
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
        self.meta_optim.zero_grad() 
        loss_q.backward() 
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
        loss = self.cross_entropy(logits, y_support)
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
            loss = self.cross_entropy(logits, y_support)
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