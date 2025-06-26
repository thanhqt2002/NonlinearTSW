import argparse

class Config:
    type = 'swae'
    batch_size = 500
    lr = 1e-3
    epochs = 100
    d = 3
    beta = 1
    device = 'cuda'  
    loss1 = 'BCE'
    loss2 = 'stsw'
    dataset = 'c10'
    prior = 'vmf'
    ntrees = 200
    nlines = 10
    delta = 2
    seed = 0


def parse_args():
    parser = argparse.ArgumentParser(description='training configs')
    parser.add_argument('--type', type=str, default='swae', choices=['ae', 'swae'], help='which ae?')
    parser.add_argument('--loss1', type=str, help='loss1', default='BCE')
    parser.add_argument('--loss2', type=str, help='loss2', default='stsw')
    parser.add_argument('--beta', type=float, help='beta', default=1)
    parser.add_argument('--d', type=int, help='embedding dim', default=3)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'c10'], help='dataset', default='c10')
    parser.add_argument('--prior', type=str, choices=['uniform', 'vmf'], help='prior', default='vmf')
    parser.add_argument('--device', type=str, help='device', default='cuda')
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ntrees', type=int, default=200)
    parser.add_argument('--nlines', type=int, default=10)
    parser.add_argument('--delta', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    return args