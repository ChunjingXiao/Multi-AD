import argparse
import torch
import time

# Set argument
parser = argparse.ArgumentParser(description='Multi-AD')

parser.add_argument('--name', type=str, default="testrun", help='Provide a test name.')

parser.add_argument('--device', type=str, default='cuda')

parser.add_argument('--dataname', default='weibo', type=str, help='The name of the dataset')

parser.add_argument('--mode', choices=['train', 'test'], default='test')

parser.add_argument('--epoch', default=500)

parser.add_argument('--lr', default=1e-6)

parser.add_argument('--dropout', default=0.3)

parser.add_argument('--log', default=False)

parser.add_argument('--hidden_channels', default=128)

parser.add_argument('--format', default='adjlist', help='File format of input file')

parser.add_argument('--undirected', default=True, type=bool, help='Treat graph as undirected.')

parser.add_argument('--number-walks', default=10, type=int, help='Number of random walks to start at each node')

parser.add_argument('--seed', default=0, type=int, help='Seed for random walk generator.')

parser.add_argument('--walk-length', default=40, type=int, help='Length of the random walk started at each node')

parser.add_argument('--representation-size', default=64, type=int, help='Number of latent dimensions to learn for each node.')

parser.add_argument('--no_cuda', action='store_false', default=True, help='Disables CUDA training.')

parser.add_argument('--seq_num', default=5)

parser.add_argument('--seq_length', default=7)


args = parser.parse_args()
args.device = torch.device('cuda:0' if args.no_cuda and torch.cuda.is_available() else 'cpu')
args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
