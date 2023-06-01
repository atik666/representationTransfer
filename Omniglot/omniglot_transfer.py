import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np
import random
import torch
from torch import nn
import argparse

class Omniglot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        
        self.transform_query = transforms.Compose([lambda x: x,
                                              transforms.Resize((self.resize, self.resize)),
                                              transforms.ToTensor(),
                                              transforms.RandomResizedCrop(self.resize,(0.8,1.0)),
                                              transforms.RandomAffine((30))
                                              ])
        
        if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
            # if root/data.npy does not exist, just download it
            self.x = transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((imgsz, imgsz)),
                                                            lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255.])


            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, 'omniglot.npy'), self.x)
            print('write into omniglot.npy.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, 'omniglot.npy'))
            print('load from omniglot.npy.')

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]
        self.x_train_aug = np.reshape(self.x_train, (self.x_train.shape[0]*self.x_train.shape[1], \
                                                self.x_train.shape[2],self.x_train.shape[3],self.x_train.shape[4]))
        self.x_train_aug = np.take(self.x_train_aug, np.random.permutation(self.x_train_aug.shape[0]),axis=0,out=self.x_train_aug)

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache_train(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
    
    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                # perm = np.random.permutation(self.n_way * self.k_shot)
                # x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize, self.resize)[perm]
                # y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                # perm = np.random.permutation(self.n_way * self.k_query)
                # x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize, self.resize)[perm]
                # y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize, self.resize)
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize, self.resize)
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache
    
    def zero_pixels(self, img):
        
        row , col = img.shape
        number_of_pixels = random.randint(row, row*col/2)

        for i in range(number_of_pixels):     
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)       
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)           
            # Color that pixel to white
            img[y_coord][x_coord] = 1
            
        return img


    def load_data_cache_train(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                #selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                selected_cls = np.random.choice(self.x_train_aug.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    # meta-training and meta-test
                    x_spt.append(self.x_train_aug[cur_class][:self.k_shot])
                    
                    query = self.x_train_aug[cur_class][:self.k_shot]                 
                    query = np.squeeze(query, axis=0) # dim: 28,28
                    
                    #query = self.zero_pixels(query) # zero pixeling
                    
                    # transform = self.transform_query(Image.fromarray(query).convert('L')) 
                    # transform = transform.cpu().detach().numpy()
                    # transform = np.expand_dims(transform, axis=1)
                    
                    transform_temp = []
                    for i in range(self.k_query):
                        transform = self.transform_query(Image.fromarray(query).convert('L')) 
                        transform = transform.cpu().detach().numpy()
                        transform = np.expand_dims(transform, axis=1)
                        transform_temp.append(transform)
                    transform = np.array(transform_temp)                     
                    
                    #print(transform.shape) # (1, 1, 28, 28)
                    
                    # TODO: run transform n times 
                    x_qry.append(transform)
                    
                    #x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])
                
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize, self.resize)
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize, self.resize) # (5,1,28,28)
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)
                #print(y_qry)

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def get_task(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        
        x_spts, y_spts, x_qrys, y_qrys = next_batch
        
        x_spts = torch.tensor(x_spts)
        y_spts = torch.tensor(y_spts)
        x_qrys = torch.tensor(x_qrys)
        y_qrys = torch.tensor(y_qrys)

        #return next_batch
        return x_spts, y_spts, x_qrys, y_qrys


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

def train(N, K, Q, dset, lr, EPISODE_NUM=1, bsize=32):
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
    
    model_emb = './saved_models/omni_emb_%dway_%dshot.pth'%(N, K)
    model_rn = './saved_models/omni_%dway_%dshot.pth'%(N, K)
    
    Emb = Embedding()
    RN = RelationNet()
    MSE = nn.MSELoss()
    
    if torch.cuda.is_available():
        print("cuda is on")
        Emb.cuda()
        RN.cuda()
        MSE=MSE.cuda()
    
    Emb_optim = torch.optim.Adam(Emb.parameters(), lr = lr)
    RN_optim = torch.optim.Adam(RN.parameters(), lr = lr)
    
    accuracies = []
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
            #print(j , spt_x[j].shape)
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
            spt_x, spt_y, qry_x, qry_y = dset.get_task('test')
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
                
            acc = (correct/float(bsize*N*Q)).detach().cpu().numpy()
            accuracies.append(acc)
            
            if  max(accuracies) == acc:
                print('\n','Saving best model...')
                torch.save(Emb.state_dict(), model_emb)
                torch.save(RN.state_dict(), model_rn)
                
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

# Parse the arguments
args = parser.parse_args()
print(args)

omniglot = Omniglot(args.path, batchsz=args.batchsz, n_way=args.n_way, 
                   k_shot=args.k_shot, k_query=args.k_query, imgsz=args.size)

train(args.n_way, args.k_shot, args.k_query, omniglot, args.lr, args.epochs, args.batchsz)
