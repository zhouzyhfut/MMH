import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tensorboardX import SummaryWriter

from data_loader import RegDBData, TestData
from data_manager import process_test_regdb
from loss import ContrastiveLoss
from model import embed_net
from paths import DATA_ROOT
from utils import AverageMeter, Logger, set_seed


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='vis_log/', type=str, help='visualization log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='data loading workers')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='CUDA device id')
parser.add_argument('--pretrain_path', default='../pretrain_clip/save_model/1.t', type=str,
                    help='CLIP-VAT checkpoint saved by pretrain_clip/pretrain.py')
parser.add_argument('--mode', default='Triplet', choices=['Triplet'], help='training mode')
args = parser.parse_args()


set_seed(args.seed)

log_path = args.log_path + f'{args.mode}/'
args.model_path += args.mode + '/'
checkpoint_path = args.model_path

for directory in (log_path, checkpoint_path, args.vis_log_path):
    if not os.path.isdir(directory):
        os.makedirs(directory)

suffix = f'base_n{args.batch_size}_lr_{args.lr}_seed_{args.seed}'

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.mode + '/' + args.vis_log_path + suffix + '/'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print('==========\nArgs:{}\n=========='.format(args))

device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Loading data..')
print('Data root: {}'.format(DATA_ROOT))
feature_to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)

end = time.time()
trainset = RegDBData(transform=feature_to_tensor)
print(len(trainset))

query_img, query_label, query_id = process_test_regdb(modal='visible')
gall_img, gall_label, gall_id = process_test_regdb(modal='thermal')
depth_img, depth_label, depth_id = process_test_regdb(modal='depth')

gallset = TestData(gall_img, gall_label, query_id, transform=feature_to_tensor)
queryset = TestData(query_img, query_label, gall_id, transform=feature_to_tensor)
depthset = TestData(depth_img, depth_label, depth_id, transform=feature_to_tensor)
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
depth_loader = data.DataLoader(depthset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)
ndepth = len(depth_label)

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
net.to(device)
ckpt = torch.load(args.pretrain_path, map_location=device)
net.imagecnn1.load_state_dict(ckpt['imagecnn1'])
net.imagetrans1.load_state_dict(ckpt['imagetrans1'])
cudnn.benchmark = True

criterion_id = nn.CrossEntropyLoss()
criterion_Cons = ContrastiveLoss()
criterion_id.to(device)
criterion_Cons.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, epoch):
    """Set the learning rate to the original schedule."""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch < 20:
        lr = args.lr
    elif epoch < 50:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for index in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[index + 1]['lr'] = lr
    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    cons_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.train()
    end = time.time()

    for batch_idx, (input1, input2, input3, label1, label2, label3, id1, id2, id3) in enumerate(trainloader):
        input1 = input1.to(device)
        input2 = input2.to(device)
        input3 = input3.to(device)
        label1 = label1.to(device)
        label2 = label2.to(device)
        label3 = label3.to(device)
        data_time.update(time.time() - end)

        out0, out1, out2, out3, feata, featt, featv = net(input1, input2, input3)
        loss_cons = (criterion_Cons(feata, featt) + criterion_Cons(feata, featv)) * 0.1

        loss_id = criterion_id(out0, label1)
        loss_video = criterion_id(out1, label1)
        loss_text = criterion_id(out2, label2)
        loss_audio = criterion_id(out3, label3)
        loss = loss_cons + loss_id + (loss_text + loss_audio + loss_video) / 3

        _, predicted = out0.max(1)
        correct += predicted.eq(label1).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), 3 * input1.size(0))
        id_loss.update(loss_id.item(), 3 * input1.size(0))
        cons_loss.update(loss_cons.item(), 3 * input1.size(0))
        total += label1.size(0)

        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print(
                'Epoch: [{}][{}/{}] '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                'lr:{:.3f} '
                'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                'CLoss: {cons_loss.val:.4f} ({cons_loss.avg:.4f}) '
                'Accu: {:.2f}'.format(
                    epoch,
                    batch_idx,
                    len(trainloader),
                    current_lr,
                    100. * correct / total,
                    batch_time=batch_time,
                    train_loss=train_loss,
                    cons_loss=cons_loss,
                    id_loss=id_loss,
                )
            )

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    net.eval()
    print('Extracting Features for Classification...')
    start = time.time()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for (input1, label1, id1), (input2, label2, id2), (input3, label3, id3) in zip(
            query_loader, gall_loader, depth_loader
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
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    print(classification_report(all_labels, all_predictions, digits=4))
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    print('==> Start Training...')
    best_epoch = -1
    best_f1 = 0
    for epoch in range(start_epoch, 81 - start_epoch):
        print('==> Preparing Data Loader...')
        print(epoch)

        trainloader = data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True,
            shuffle=True,
        )

        train(epoch)

        if epoch % 1 == 0:
            print('Test Epoch: {}'.format(epoch))
            accuracy, precision, recall, f1 = test(epoch)

            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = epoch
                state = {
                    'net': net.state_dict(),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_best.t')

            if f1 > best_f1:
                best_f1 = f1
                best_epoch1 = epoch
                state = {
                    'net': net.state_dict(),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + suffix + '_bestf1.t')

            print(f'Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1-Score: {f1:.2%}')
            print('Best Epoch [{}]'.format(best_epoch))
            print('Best f1Epoch [{}]'.format(best_epoch1))
