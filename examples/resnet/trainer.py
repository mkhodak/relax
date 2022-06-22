import argparse
import json
import math
import os
import pdb
import shutil
import time
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
import resnet

from relax.nas import MixedOptimizer, Supernet
from relax.xd import fixed, original


class RowColPermute(nn.Module):

    def __init__(self, row, col):

        super().__init__()
        try:
            from torch_butterfly.permutation import bitreversal_permutation
            self.rowperm = torch.LongTensor(bitreversal_permutation(row))
            self.colperm = torch.LongTensor(bitreversal_permutation(col))
            print("Using bit-reversal permutation")
        except ImportError:
            self.rowperm = torch.randperm(row) if type(row) == int else row
            self.colperm = torch.randperm(col) if type(col) == int else col
            print("Using random permutation")

    def forward(self, tensor):

        return tensor[:,self.rowperm][:,:,self.colperm]


model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--backbone', type=str, default='resnet20')
parser.add_argument('--data', default='cifar10', type=str)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='results', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--arch-lr', default=0.1, type=float)
parser.add_argument('--arch-adam', action='store_true')
parser.add_argument('--xd', action='store_true')
parser.add_argument('--fft', action='store_true')
parser.add_argument('--compact', action='store_true')
parser.add_argument('--einsum', action='store_true')
parser.add_argument('--kmatrix-depth', default=1, type=int)
parser.add_argument('--warmup-epochs', default=0, type=int)
parser.add_argument('--permute', action='store_true')
parser.add_argument('--get-permute', type=str, default='')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    model = resnet.__dict__[args.backbone](num_classes=int(args.data[5:]))
    torch.cuda.set_device(args.device)
    
    criterion = nn.CrossEntropyLoss().cuda()
    writer = SummaryWriter(args.save_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.permute or args.get_permute:
        if args.get_permute:
            permute = torch.load(args.get_permute)['permute']
        elif args.resume:
            permute = torch.load(args.resume)['permute']
        else:
            permute = RowColPermute(32, 32)
        train_transforms = [transforms.ToTensor(), permute, normalize]
        val_transforms = [transforms.ToTensor(), permute, normalize]
    else:
        permute = None
        train_transforms = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize]
        val_transforms = [transforms.ToTensor(), normalize]

    cifar = datasets.CIFAR100 if args.data == 'cifar100' else datasets.CIFAR10
    train_loader = torch.utils.data.DataLoader(
        cifar(root='./data', train=True, transform=transforms.Compose(train_transforms), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        cifar(root='./data', train=False, transform=transforms.Compose(val_transforms)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.half:
        model.half()
        criterion.half()

    if args.fft:
        Supernet.create(model, in_place=True)
        X, _ = next(iter(train_loader))
        arch_kwargs = {'arch': fixed,
                       'compact': args.compact,
                       'einsum': args.einsum,
                       'verbose': not args.resume}
        model.conv2xd(X[:1], **arch_kwargs)
    if not args.xd:
        args.arch_lr = 0.0
    print('Model weight count:', sum(p.numel() for p in model.parameters()))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    model.cuda()

    # define optimizer
    momentum = partial(torch.optim.SGD, momentum=args.momentum)
    optimizer = momentum(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def sched(epoch):

        if epoch < 1 and args.backbone in ['resnet1202', 'resnet110']:
            return 0.1
        return 0.1 ** (epoch >= int(0.5 * args.epochs)) * 0.1 ** (epoch >= int(0.75 * args.epochs))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched, last_epoch=-1)
    for epoch in range(args.start_epoch):
        optimizer.step() and lr_scheduler.step()

    if not args.evaluate:
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    for epoch in range(args.start_epoch, args.epochs):

        if args.xd and (epoch == args.warmup_epochs or (args.resume and epoch == args.start_epoch and epoch >= args.warmup_epochs)):
            model.cpu()
            Supernet.create(model, in_place=True)
            X, _ = next(iter(train_loader))
            arch_kwargs = {'arch': original,
                           'compact': args.compact,
                           'einsum': args.einsum,
                           'depth': args.kmatrix_depth,
                           'verbose': not args.resume}
            model.conv2xd(X[:1], **arch_kwargs)
            print('Arch param count:', sum(p.numel() for p in model.arch_params()))
            model.cuda()
            arch_opt = torch.optim.Adam if args.arch_adam else momentum
            optimizer = MixedOptimizer([momentum(model.model_weights(), lr=args.lr, weight_decay=args.weight_decay),
                                        arch_opt(model.arch_params(), lr=args.arch_lr, weight_decay=0.0 if args.arch_adam else args.weight_decay)])
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched, last_epoch=epoch-1)

        if args.resume and epoch == args.start_epoch:
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state'])

        if args.evaluate:
            validate(val_loader, model, criterion)
            return

        writer.add_scalar('hyper/lr', optimizer.param_groups[0]['lr'], epoch)
        if args.xd:
            writer.add_scalar('hyper/arch', 0.0 if len(optimizer.param_groups) == 1 else optimizer.param_groups[1]['lr'], epoch)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        acc, loss = train(train_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('train/acc', acc, epoch)
        writer.add_scalar('train/loss', loss, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, loss = validate(val_loader, model, criterion)
        writer.add_scalar('valid/acc', prec1, epoch)
        writer.add_scalar('valid/loss', loss, epoch)

        # remember best prec@1 and save checkpoint
        best_prec1 = max(prec1, best_prec1)

        model.train()
        if (epoch+1) % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_prec1': best_prec1,
                'permute': permute,
            }, os.path.join(args.save_dir, 'checkpoint.th'))
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'permute': permute,
        }, os.path.join(args.save_dir, 'model.th'))

    try:
        model.save_arch(os.path.join(args.save_dir, 'arch.th'))
    except AttributeError:
        pass
    writer.flush()
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({'final validation accuracy': prec1,
                   'best validation accuracy': best_prec1,
                   }, f, indent=4)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

def save_checkpoint(state, filename):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':

    main()
