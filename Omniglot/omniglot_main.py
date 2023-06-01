from omni_sup_class import OmniglotNShot
import torch
import numpy as np
from meta import Meta
from tqdm import tqdm
import argparse

def main():
    
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Few-shot MAML on Omniglot for unsupervised representation learning."
        )
    
    # Add the parameters
    parser.add_argument('-p', '--path', type=str, default='/home/atik/Documents/MAML/Summer_1/datasets/Omniglot/', 
                        help='Path to unsupervised training data')
    parser.add_argument('-model', '--model_path', type=str, default='./saved_models/', 
                        help='Path to test images')
    parser.add_argument('-n', '--n_way', type=int, default=5, help='Number of ways')
    parser.add_argument('-s', '--k_shot', type=int, default=1, help='Number of support set')
    parser.add_argument('-q', '--k_query', type=int, default=1, help='Number of query set')
    parser.add_argument('-e', '--epochs', type=int, default=40000, help='Number of  epochs')
    parser.add_argument('-b', '--batchsz', type=int, default=32, help='Batch size')
    parser.add_argument('-sz', '--size', type=int, default=28, help='Image size')
    parser.add_argument('-t', '--transfer', type=eval, default='True', help='If true, applies transfer learning, else regular MAML')
    parser.add_argument('-r', '--seed', type=int, default=222, help='Random seed')
    
    # Parse the arguments
    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)


    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device('cuda:0')
    maml = Meta(config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print("Model: \n", maml)
    print('Total trainable tensors:', num)
    
    if args.transfer == True:
        try:
            model_path = args.model_path + 'model_%sw_%ss_%sq.pth' %(args.n_way,args.k_shot,args.k_query)
            maml.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print("File does not exist. Please save the file first by running the relevant unsupervised learning first.\n")
            raise          
    else:
        pass
    
    db_train = OmniglotNShot(args.path,
                   batchsz=args.batchsz,
                   n_way=args.n_way,
                   k_shot=args.k_shot,
                   k_query=args.k_query,
                   imgsz=args.size)
    
    accuracies = []
    for step in tqdm(range(args.epochs)):
        
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry)
        
        if step % 50 == 0:
            print('\n','step:', step, '\ttraining acc:', accs)

        if step % 100 == 0:  # evaluation

            accs = []
            for _ in range(1000//args.batchsz):
            # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            accuracies.append(accs[-1])
            
            print('\n','Test acc:', accs)
            print("Best accuracy: ", max(accuracies))
    
    print("\n", "Best test accuracy: ", max(accuracies))

main()    
