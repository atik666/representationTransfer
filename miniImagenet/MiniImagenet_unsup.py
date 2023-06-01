import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import random
from os import walk
import glob

class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all images
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, 
    especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize):
        """
        :param startidx: start to index label from startidx
        """
        self.mode = mode
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.s = 0.5
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 # transforms.RandomHorizontalFlip(),
                                                 # transforms.RandomRotation(5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
            self.transform_query = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                  transforms.RandomHorizontalFlip(0.5),
                                                  transforms.RandomResizedCrop(self.resize,(0.8,1.0)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                  transforms.RandomInvert(p=0.5),
                                                  transforms.GaussianBlur(kernel_size=9),
                                                  transforms.RandomApply(
                                                    [transforms.ColorJitter(
                                                      0.8*self.s, 
                                                      0.8*self.s, 
                                                      0.8*self.s, 
                                                      0.2*self.s)], p = 0.8),
                                                    transforms.RandomGrayscale(p=0.2)])
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        if self.mode == "train":
            self.path = root
            self.img = self.loadCSV_train() 
            self.create_batch_train()
        elif self.mode == 'test':         
            self.path = os.path.join(root, mode,"")  # image path
            dictLabels = self.loadCSV()  # csv path
        
            self.data = []
            #self.img2label = {}
            for i, (label, imgs) in enumerate(dictLabels.items()):
                self.data.append(imgs)  # [[img1, img2, ...], [img111, ...]]
                #self.img2label[label] = i  # {"img_name[:9]":label}
            self.cls_num = len(self.data)
    
            self.create_batch()
        
    def loadCSV_train(self):
        
        filenames = next(walk(self.path))[2]

        img = []
        for i in range(len(filenames)):  
            for images in glob.iglob(f'{self.path+filenames[i]}'):
                # check if the image ends with png
                if (images.endswith(".jpeg")) or (images.endswith(".jpg")):
                    img_temp = images[len(self.path):]
                    img.append(img_temp)
        
        return img

    def loadCSV(self):
        
        filenames = next(walk(self.path))[1]
    
        dictLabels = {}
        
        for i in range(len(filenames)):  
            img = []
            for images in glob.iglob(f'{self.path+filenames[i]}/*'):
                # check if the image ends with png
                if (images.endswith(".jpeg")) or (images.endswith(".jpg")):
                    img_temp = images[len(self.path+filenames[i]+'/'):]
                    img_temp = filenames[i]+'/'+img_temp
                    img.append(img_temp)
                
                dictLabels[filenames[i]] = img
                    
        return dictLabels

    def create_batch(self):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.selected_classes = []
        for b in range(self.batchsz):  # for each batch
            # 1.select n_way classes randomly
        
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            selected_classes_temp = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())
                selected_classes_temp.append(cls)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
            self.selected_classes.append(selected_classes_temp)
            
    def create_batch_train(self):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        for _ in range(self.batchsz):  # for each batch
            # 1.select n_way classes randomly
        
            selected_cls = np.arange(self.n_way)
            np.random.shuffle(selected_cls)
            
            selected_imgs_idx = np.random.choice(len(self.img), self.n_way, False)
            support_x = [self.img[selected_imgs_idx[i]] for i in range(len(selected_imgs_idx))]

            self.support_x_batch.append(support_x)  # append set to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)

        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        
        if self.mode == 'train':
        
            flatten_support_x = [os.path.join(self.path, self.support_x_batch[index][item])
                                         for item in range(len(self.support_x_batch[index]))]
            
            flatten_query_x = np.repeat(flatten_support_x,self.k_query).tolist()
            
            for i, path in enumerate(flatten_support_x):
                support_x[i] = self.transform(path)
            for i, path in enumerate(flatten_query_x):
                query_x[i] = self.transform_query(path)
            
            support_y = np.arange(self.n_way)
            np.random.shuffle(support_y)
            
            query_y = np.repeat(support_y,self.k_query)
            
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)
        
        elif self.mode == 'test':       

            flatten_support_x = [os.path.join(self.path, item)
                                 for sublist in self.support_x_batch[index] for item in sublist]

            support_y_list = []
            for i in range(len(self.support_x_batch[index])):
                class_temp = np.repeat(self.selected_classes[index][i], len(self.support_x_batch[index][i]))
                support_y_list.append(class_temp)
            support_y = np.array(support_y_list).flatten().astype(np.int32)
            #print("support_y: ", support_y)
    
            flatten_query_x = [os.path.join(self.path, item)
                               for sublist in self.query_x_batch[index] for item in sublist]

            query_y_list = []
            for i in range(len(self.query_x_batch[index])):
                class_temp = np.repeat(self.selected_classes[index][i], len(self.query_x_batch[index][i]))
                query_y_list.append(class_temp)
            query_y = np.array(query_y_list).flatten().astype(np.int32)
            #print("query_y: ", query_y)
    
            # unique: [n-way], sorted
            unique = np.unique(support_y)
            random.shuffle(unique)
            # relative means the label ranges from 0 to n-way
            support_y_relative = np.zeros(self.setsz)
            query_y_relative = np.zeros(self.querysz)
            for idx, l in enumerate(unique):
                support_y_relative[support_y == l] = idx
                query_y_relative[query_y == l] = idx
    
            # print('relative:', support_y_relative, query_y_relative)
    
            for i, path in enumerate(flatten_support_x):
                support_x[i] = self.transform(path)
    
            for i, path in enumerate(flatten_query_x):
                    query_x[i] = self.transform(path)
    
            return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    import time
    
    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    
    mini = MiniImagenet('./dataset/unsupervised/', 
                     mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=224)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_
        support_x = make_grid(support_x, nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)

        tb.add_image('support_x', support_x)
        tb.add_image('query_x', query_x)

        time.sleep(5)

    tb.close()
