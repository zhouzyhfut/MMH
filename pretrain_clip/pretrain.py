from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import clip
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from pretrainclipmodel import CLIPVAT
from data_loader import CLIPdataset1
from paths import DATA_ROOT
from utils import *
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--weight_decay', default=1e-3, type=float)
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
log_path = args.log_path + 'regdb_log/'
test_mode = [2, 1, 3] 

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)



sys.stdout = Logger(
    log_path + f'.txt')

vis_log_dir = args.vis_log_path + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0

print('==> Loading data..')
print('Data root: {}'.format(DATA_ROOT))
transform_feat = lambda x: torch.tensor(x, dtype=torch.float32)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

end = time.time()
dataset == 'regdb'
trainset = CLIPdataset1(transform=transform_feat)
print(len(trainset))  
n_class = len(np.unique(trainset.train_color_id))
print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # features')
print('  ------------------------------')
print('  video    | {:5d} | {:8d}'.format(n_class, len(trainset)))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net =CLIPVAT()
net.to(device)

if len(args.resume) > 0:
    model_path = args.resume
    if not os.path.isabs(model_path):
        model_path = os.path.join(checkpoint_path, model_path)
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location=device)
        net.imagecnn1.load_state_dict(checkpoint['imagecnn1'])
        net.imagetrans1.load_state_dict(checkpoint['imagetrans1'])
        start_epoch = checkpoint['epoch'] + 1
        print('==> loaded checkpoint {} (epoch {})'
              .format(model_path, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(model_path))

criterion_id = nn.CrossEntropyLoss()
criterion_id.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01
    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0
    # switch to train mode
    net.train()
    end = time.time()
    # print(99887, net)
    for batch_idx, (video1,text) in enumerate(trainloader):
        # print(target3,text)
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        cliptext = clip.tokenize(text).to(device)
        video1=video1.to(device)
        logit1,logit2=net(video1,cliptext)
        targets = torch.arange(len(video1), dtype=torch.long, device=device)
        loss = (criterion_id(logit1, targets) + criterion_id(logit2, targets)) / 2
        _, predicted = logit1.max(1)
        correct += (predicted.eq(targets).sum().item())

        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), video1.size(0))
        total += targets.size(0)
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 20 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'Accu: {:.2f}'
            .format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss,
                id_loss=id_loss))
    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    return 100. * correct / total



# training
if __name__ == "__main__":
    print('==> Start Training...')
    best_epoch = -1
    best_acc = 0
    loader_batch = args.batch_size
    trainloader = data.DataLoader(trainset, batch_size=loader_batch, drop_last=True, shuffle=True)
    for epoch in range(start_epoch, 80):
        print('==> Preparing Data Loader...')
        print(epoch)
        # training
        acc=train(epoch)
        if epoch % 1 == 0:
            # if epoch > 0 and epoch % 2 == 0:
            print('Test Epoch: {}'.format(epoch))
            if acc>best_acc: 
                best_epoch = epoch
                state = {
                    'imagecnn1': net.imagecnn1.state_dict(),
                    'imagetrans1': net.imagetrans1.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, checkpoint_path + f'{epoch}.t')
