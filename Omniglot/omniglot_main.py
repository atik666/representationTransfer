import torch
from torch import nn
import torchvision
import torchvision.transforms as transform
import numpy as np
import argparse

class Omniglot:
    def __init__(self, root, bsize=32, N=5, K=5, Q=15, imgsz=28):
        """
        params:
            bsize = batch size
            N = N-way
            K = K-shot
            Q = Num of query
        """
        assert K+Q <= 20, "num of K + num of Q should be less than 20"
        self.bsize=bsize
        self.N=N
        self.K=K
        self.Q=Q
        self.root = root
        self.imgsz = imgsz
        
        self.dtrain=torchvision.datasets.Omniglot(
            self.root, background=True, download=True, 
            transform = transform.Compose([transform.Resize([self.imgsz,self.imgsz], interpolation=2), 
                                               transform.ToTensor()])
        )
        self.dtest=torchvision.datasets.Omniglot(
            self.root, background=False, download=True, 
            transform = transform.Compose([transform.Resize([self.imgsz,self.imgsz], interpolation=2), 
                                               transform.ToTensor()])
        )
        
        self.data_num = len(self.dtrain)+len(self.dtest) 
        self.cls_num = int(self.data_num/20)
                           
        print('Num of total cls :', self.cls_num)
        print('Num of total data :',  self.data_num)
        
    def get_task(self, mode='train'):
        """
        params:
            mode : 'train' or 'test'
        """
        if mode=='train':
            dset = self.dtrain
            cls_num = int(len(self.dtrain)/20)
        else:
            dset = self.dtest
            cls_num = int(len(self.dtest)/20)
            
        spt_xs=torch.zeros([self.bsize, self.N*self.K, self.imgsz, self.imgsz])
        spt_ys=torch.zeros([self.bsize, self.N*self.K], dtype=torch.int64)
        qry_xs=torch.zeros([self.bsize, self.N*self.Q, self.imgsz, self.imgsz])
        qry_ys=torch.zeros([self.bsize, self.N*self.Q], dtype=torch.int64)
        
        for i in range(self.bsize):
            n_way = np.random.choice(cls_num, self.N, replace=False)
            
            spt_x=torch.zeros([self.N, self.K,self.imgsz,self.imgsz])
            spt_y=torch.zeros([self.N, self.K])
            qry_x=torch.zeros([self.N, self.Q,self.imgsz,self.imgsz])
            qry_y=torch.zeros([self.N, self.Q])
             
            for j, idx in enumerate(n_way):
                spt_x_, _ = zip(*[dset[i] for i in range(idx*20, idx*20+self.K)])
                spt_x[j] = torch.stack(spt_x_).resize(self.K,self.imgsz,self.imgsz)
                spt_y[j] = torch.tensor([j for k in range(self.K)])
                qry_x_, _ = zip(*[dset[i] for i in range(idx*20+self.K, idx*20+self.K+self.Q)])
                qry_x[j] = torch.stack(qry_x_).resize(self.Q,self.imgsz,self.imgsz)
                qry_y[j] = torch.tensor([j for k in range(self.Q)])
            
            spt_xs[i] = spt_x.reshape(self.N * self.K, self.imgsz, self.imgsz)
            spt_ys[i] = spt_y.reshape(self.N * self.K)
            qry_xs[i] = qry_x.reshape(self.N * self.Q, self.imgsz, self.imgsz)
            qry_ys[i] = qry_y.reshape(self.N * self.Q)
        
        return spt_xs, spt_ys, qry_xs, qry_ys
    
class Embedding(nn.Module):
    def __init__(self):
        
        nn.Module.__init__(self)
        
        in_channel = 1
        out_channel = 64
                 
        layers = []
        for i in range(4):
            if i<2:
                layers += [nn.Conv2d(in_channel, out_channel, 3, 1, 0), nn.BatchNorm2d(out_channel), nn.ReLU(), nn.MaxPool2d(2)]
            else:
                layers += [nn.Conv2d(in_channel, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), nn.ReLU()]
            in_channel=out_channel
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(-1 ,1, 28, 28)
        return self.layers(x)
    
class RelationNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        
        layers= []
        layers += [nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)]
        layers += [nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)]
        
        self.layer1 = nn.Sequential(*layers)
        layers= []
        layers += [nn.Linear(64,8), nn.ReLU()]
        layers += [nn.Linear(8,1), nn.Sigmoid()]
        
        self.layer2 = nn.Sequential(*layers)
    
    def forward(self, x):
        x=x.view(-1,128,5,5)
        x=self.layer1(x).reshape(-1,64)
        return self.layer2(x)
    
def train(N, K, Q, dset, lr, EPISODE_NUM=1, bsize=32, transfer=True):
    """
    parameters:
        N : N way
        K : K shot
        Q : Num of query data for each class
        dset : dataset(omniglot)
        lr : learning rate
        EPISODE_NUM = episode_num
        bsize = batch szie
    """
    Emb = Embedding()
    RN = RelationNet()
    MSE = nn.MSELoss()
    
    model_emb = './saved_models/omni_emb_%dway_%dshot.pth'%(N, K)
    model_rn = './saved_models/omni_%dway_%dshot.pth'%(N, K)
    
    if transfer:
        Emb.load_state_dict(torch.load(model_emb))
        RN.load_state_dict(torch.load(model_rn))
        
    if torch.cuda.is_available():
        print("cuda is on")
        Emb.cuda()
        RN.cuda()
        MSE=MSE.cuda()
    
    Emb_optim = torch.optim.Adam(Emb.parameters(), lr = lr)
    RN_optim = torch.optim.Adam(RN.parameters(), lr = lr)
    
    
    for i in range(EPISODE_NUM):
        Emb.train()
        RN.train()
        spt_x, spt_y, qry_x, qry_y = dset.get_task('train')
        spt_x = torch.autograd.Variable(spt_x)
        spt_y = torch.autograd.Variable(spt_y)
        qry_x = torch.autograd.Variable(qry_x)
        qry_y = torch.autograd.Variable(qry_y) 
        
        if torch.cuda.is_available(): 
            spt_x=spt_x.cuda()
            spt_y=spt_y.cuda()
            qry_x=qry_x.cuda()
            qry_y=qry_y.cuda()
            
        for j in range(bsize):
            spt_emb = Emb(spt_x[j]).reshape(N, K, 64, 5, 5).sum(1).unsqueeze(0).repeat(N*Q,1,1,1,1)
            qry_emb = Emb(qry_x[j]).unsqueeze(0).repeat(N,1,1,1,1).transpose(0,1)
            concat = torch.cat((spt_emb, qry_emb), 2)
            score = RN(concat).reshape(N*Q, N)
            
            one_hot = torch.zeros(N*Q, N)
            one_hot[torch.arange(N*Q), qry_y[j]] = 1
            one_hot = torch.autograd.Variable(one_hot)
            if torch.cuda.is_available(): 
                one_hot=one_hot.cuda()
            
            loss = MSE(score, one_hot)
            Emb.zero_grad()
            RN.zero_grad()

            loss.backward()
            
            Emb_optim.step()
            RN_optim.step()
        
        if i%10 == 0 :
            spt_x, spt_y, qry_x, qry_y = dset.get_task('train')
            Emb.eval()
            RN.eval()
            
            if torch.cuda.is_available(): 
                spt_x=spt_x.cuda()
                spt_y=spt_y.cuda()
                qry_x=qry_x.cuda()
                qry_y=qry_y.cuda()
                one_hot=one_hot.cuda()
                
            correct=0

            for j in range(bsize):
                spt_emb = Emb(spt_x[j]).reshape(N, K, 64, 5, 5).sum(1).unsqueeze(0).repeat(N*Q,1,1,1,1)
                qry_emb = Emb(qry_x[j]).unsqueeze(0).repeat(N,1,1,1,1).transpose(0,1)
                concat = torch.cat((spt_emb, qry_emb), 2)
                score = RN(concat).reshape(N*Q, N)
                pred = score.max(1)[1]
                correct = correct + torch.sum(pred==qry_y[j])
                
            print("Epoch ",i, ":")
            print("accuracy after training : ", correct/float(bsize*N*Q))
    
    
# Initialize the parser
parser = argparse.ArgumentParser(
    description="Few-shot RN for unsupervised representation learning on Ominglot."
    )

# Add the parameters
parser.add_argument('-p', '--path', type=str, default='./dataset/', 
                    help='Path to the unsupervised images')
parser.add_argument('-n', '--n_way', type=int, default=5, help='Number of ways')
parser.add_argument('-s', '--k_shot', type=int, default=1, help='Number of support set')
parser.add_argument('-q', '--k_query', type=int, default=1, help='Number of query set')
parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of  epochs')
parser.add_argument('-b', '--batchsz', type=int, default=32, help='dataloader batch size')
parser.add_argument('-sz', '--size', type=int, default=28, help='Image size')
parser.add_argument('-l', '--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('-t', '--transfer', type=eval, default='True', help='If true, applies transfer learning, else regular RN')

# Parse the arguments
args = parser.parse_args()
print(args)

omniglot = Omniglot(args.path, bsize=args.batchsz, N=args.n_way, 
                   K=args.k_shot, Q=args.k_query, imgsz=args.size)

train(args.n_way, args.k_shot, args.k_query, omniglot, args.lr, args.epochs, args.batchsz, args.transfer)