import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import random
from os import walk
import glob
from meta import Meta
from    torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

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
            # :return: dictLabels: {label1: [filename1, filename2, filename3, filename4,...], }
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

            # shuffle the correponding relation between support set and query set
            # random.shuffle(support_x)
            # random.shuffle(query_x)

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
        # [setsz]
        #support_y = np.zeros((self.setsz), dtype=np.int32)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        #query_y = np.zeros((self.querysz), dtype=np.int32)
        
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
    
            # print('global:', support_y, query_y)
            # support_y: [setsz]
            # query_y: [querysz]
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

def main():
    
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Few-shot MAML for unsupervised representation learning."
        )
    
    # Add the parameters
    parser.add_argument('-train', '--train_path', type=str, default='./dataset/unsupervised/', 
                        help='Path to training images')
    parser.add_argument('-test', '--test_path', type=str, default='./dataset/', 
                        help='Path to test images')
    parser.add_argument('-model', '--model_path', type=str, default='./saved_models/', 
                        help='Path to test images')
    parser.add_argument('-n', '--n_way', type=int, default=20, help='Number of ways')
    parser.add_argument('-s', '--k_shot', type=int, default=1, help='Number of support set')
    parser.add_argument('-q', '--k_query', type=int, default=1, help='Number of query set')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of  epochs')
    parser.add_argument('-t', '--temp', type=int, default=10, help='temperature')
    parser.add_argument('-b_train', '--batchsz_train', type=int, default=10000, help='Train batch size')
    parser.add_argument('-b_test', '--batchsz_test', type=int, default=100, help='Train batch size')
    parser.add_argument('-sz', '--size', type=int, default=84, help='Image size')
    parser.add_argument('-r', '--seed', type=int, default=222, help='Random seed')
    
    # Parse the arguments
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda:0')
    if args.n_way == 5:
        args.temp = 100
    elif args.n_way == 20:
        args.temp = 10
        
    maml = Meta(config, args.temp).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print("Model: \n", maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number

    model_path = args.model_path + 'model_%sw_%ss_%sq.pth' %(args.n_way,args.k_shot,args.k_query)
    opt_path = args.model_path + 'opt_%sw_%ss_%sq.pth' %(args.n_way,args.k_shot,args.k_query)
    
    mini_train = MiniImagenet(args.train_path, mode='train', n_way=args.n_way, k_shot=args.k_shot,
                        k_query=args.k_query,
                        batchsz=args.batchsz_train, resize=args.size)
    mini_test = MiniImagenet(args.test_path, mode='test', n_way=args.n_way, k_shot=args.k_shot,
                             k_query=args.k_query,
                             batchsz=args.batchsz_test, resize=args.size)
    
    accuracies = []
    for epoch in tqdm(range(args.epochs)):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini_train, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry)
            
            if step % 100 == 0:
                print('\n','step:', step, '\ttraining acc:', accs)

            if step % 1000 == 0:  # evaluation
                db_test = DataLoader(mini_test, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
                accs_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                accuracies.append(accs[-1])
                print('\n','Test acc:', accs)
                print("Best accuracy: ", max(accuracies))
                
                if  max(accuracies) == accs[-1]:
                    print('\n','Saving best model...')
                    torch.save(maml.state_dict(), model_path)
                    optimizer = maml.optimizer()
                    torch.save(optimizer.state_dict(), opt_path)
        
        print("\n", "Best test accuracy: ", max(accuracies))

if __name__ == '__main__':
    main()    
