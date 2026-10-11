import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from data_loader import RegDBData, TestData
from data_manager import process_test_regdb
from model import embed_net
from paths import DATA_ROOT
from utils import Logger, set_seed


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Testing')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--log_path', default='testlog/', type=str, help='log save path')
parser.add_argument('--load_path', default='', type=str,
                    help='checkpoint produced by train.py (required)')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='data loading workers')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='CUDA device id')
parser.add_argument('--pretrain_path', default='../pretrain_clip/save_model/1.t', type=str,
                    help='CLIP-VAT checkpoint saved by pretrain_clip/pretrain.py')
parser.add_argument('--mode', default='Triplet', choices=['Triplet'], help='test mode')
args = parser.parse_args()

if not args.load_path:
    parser.error('--load_path is required: point it to a checkpoint produced by train.py')


set_seed(args.seed)

log_path = args.log_path + f'{args.mode}/'
if not os.path.isdir(log_path):
    os.makedirs(log_path)

suffix = f'mode{args.mode}_lr_{args.lr}_seed_{args.seed}'
sys.stdout = Logger(log_path + suffix + '_os.txt')
print('==========\nArgs:{}\n=========='.format(args))

device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

print('==> Loading data..')
print('Data root: {}'.format(DATA_ROOT))
feature_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)

end = time.time()
trainset = RegDBData(transform=feature_to_tensor)
print(len(trainset))

query_img, query_label, query_id = process_test_regdb(
    modal='visible',
    mode='test',
)
gall_img, gall_label, gall_id = process_test_regdb(
    modal='thermal',
    mode='test',
)
depth_img, depth_label, depth_id = process_test_regdb(
    modal='depth',
    mode='test',
)

gallset = TestData(gall_img, gall_label, query_id, transform=feature_to_tensor)
queryset = TestData(query_img, query_label, gall_id, transform=feature_to_tensor)
depthset = TestData(depth_img, depth_label, depth_id, transform=feature_to_tensor)
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
depth_loader = data.DataLoader(depthset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset statistics:')
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(n_class, dataset=trainset)
ckpt = torch.load(args.pretrain_path, map_location=device)
net.imagecnn1.load_state_dict(ckpt['imagecnn1'])
net.imagetrans1.load_state_dict(ckpt['imagetrans1'])
ckpt = torch.load(args.load_path, map_location=device)
net.load_state_dict(ckpt['net'])
net.to(device)
cudnn.benchmark = True


def test(epoch):
    net.eval()
    print('Extracting Features for Classification...')
    start = time.time()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for (input1, label1, id1), (input2, label2, id2), (input3, label3, id3) in zip(
            query_loader,
            gall_loader,
            depth_loader,
        ):
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            labels = label1.to(device)

            out0, out1, out2, out3, feata, featt, featv = net(input1, input2, input3)

            _, predicted = out0.max(1)
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    print('Feature Extraction Time: {:.3f} seconds'.format(time.time() - start))

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(classification_report(all_labels, all_predictions, digits=4))
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    print('==> Start Training...')
    for epoch in range(1):
        print('==> Preparing Data Loader...')
        print(epoch)
        print('Test Epoch: {}'.format(epoch))
        accuracy, precision, recall, f1 = test(epoch)
        print(f'Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1-Score: {f1:.2%}')
        print('Best ACC [{}]'.format(accuracy))
        print('Best f1Epoch [{}]'.format(f1))
