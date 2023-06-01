import torch, os
import numpy as np
from torch import optim
from torch.autograd import Variable
from MiniImagenet_unsup import MiniImagenet
from compare import Compare
from utils import make_imgs
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse

if __name__ == '__main__':
    
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Few-shot RN for unsupervised representation learning."
        )
    
    # Add the parameters
    parser.add_argument('-train', '--train_path', type=str, default='./dataset/unsupervised/', 
                        help='Path to training images')
    parser.add_argument('-test', '--test_path', type=str, default='./dataset/', 
                        help='Path to test images')
    parser.add_argument('-n', '--n_way', type=int, default=5, help='Number of ways')
    parser.add_argument('-s', '--k_shot', type=int, default=1, help='Number of support set')
    parser.add_argument('-q', '--k_query', type=int, default=1, help='Number of query set')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of  epochs')
    parser.add_argument('-b_train', '--batchsz_train', type=int, default=10000, help='Train batch size')
    parser.add_argument('-b_test', '--batchsz_test', type=int, default=100, help='Train batch size')
    parser.add_argument('-b', '--batchsz', type=int, default=3, help='dataloader batch size')
    parser.add_argument('-sz', '--size', type=int, default=224, help='Image size')
    parser.add_argument('-g', '--gpu', type=list, default=[0], help='GPU ids. Expand if necessary: [0,1,2,3,4,5]')
    
    # Parse the arguments
    args = parser.parse_args()
    print(args)

    n_way = args.n_way
    k_shot = args.k_shot
    k_query = args.k_query # query num per class
    batchsz = args.batchsz
	# Multi-GPU support
    print('To run on single GPU, change device_ids=[0] and downsize batch size! \n mkdir ckpt if not exists!')
    net = torch.nn.DataParallel(Compare(args.n_way, args.k_shot), device_ids=args.gpu).cuda()
	# print(net)
    if (not os.path.exists("ckpt")):
        os.mkdir("ckpt")
    mdl_file = 'ckpt/compare%d%d_unsup.mdl'%(args.n_way, args.k_shot)

    # remove if GPU full
# 	if os.path.exists(mdl_file):
# 		print('load checkpoint ...', mdl_file)
# 		net.load_state_dict(torch.load(mdl_file))

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('total params:', params)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    tb = SummaryWriter('runs', str(datetime.now()))  
    
    mini = MiniImagenet(args.train_path, mode='train', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query,
		                    batchsz=args.batchsz_train, resize=args.size)
    
    mini_val = MiniImagenet(args.test_path, mode='test', n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query,
		                        batchsz=args.batchsz_test, resize=args.size)

    best_accuracy = 0
    for epoch in range(args.epochs):

        db = DataLoader(mini, batchsz, shuffle=True, num_workers=8, pin_memory=True)

        db_val = DataLoader(mini_val, batchsz, shuffle=True, num_workers=2, pin_memory=True)

        for step, batch in enumerate(db):
            support_x = Variable(batch[0]).cuda()
            support_y = Variable(batch[1]).cuda()
            query_x = Variable(batch[2]).cuda()
            query_y = Variable(batch[3]).cuda()

            net.train()
            loss = net(support_x, support_y, query_x, query_y)
            loss = loss.mean() # Multi-GPU support

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_val_loss = 0
            if step % 200 == 0:
                total_correct = 0
                total_num = 0
                display_onebatch = False # display one batch on tensorboard
                for j, batch_test in enumerate(db_val):
                    support_x = Variable(batch_test[0]).cuda()
                    support_y = Variable(batch_test[1]).cuda()
                    query_x = Variable(batch_test[2]).cuda()
                    query_y = Variable(batch_test[3]).cuda()

                    net.eval()
                    pred, correct = net(support_x, support_y, query_x, query_y, False)
                    correct = correct.sum() # multi-gpu support
                    #total_correct += correct.data[0]
                    total_correct += correct.data
                    total_num += query_y.size(0) * query_y.size(1)

                    if not display_onebatch:
                        display_onebatch = True  # only display once
                        all_img, max_width = make_imgs(n_way, k_shot, k_query, support_x.size(0),
						                               support_x, support_y, query_x, query_y, pred)
                        all_img = make_grid(all_img, nrow=max_width)
                        tb.add_image('result batch', all_img)

                accuracy = total_correct / total_num
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(net.state_dict(), mdl_file)
                    print('saved to checkpoint:', mdl_file)

                tb.add_scalar('accuracy', accuracy)
                print('<<<<>>>>accuracy:', accuracy.cpu().detach().numpy(), 'best accuracy:', best_accuracy.cpu().detach().numpy())

            if step % 15 == 0 and step != 0:
                tb.add_scalar('loss', loss.cpu().data)
                print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%f' % (
                    n_way, k_shot, batchsz, epoch, step, loss.cpu().data))
                