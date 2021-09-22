import numpy as np
import shutil
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config as cfg


class AverageMeter(object):           
    """ Computes and stores the average and current value """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def logger(info, file_path=cfg.log_path, flag=True, init=False):
    
    if init:
        with open(file_path, 'w') as fo:
            pass
        return
    
    if flag:
        print(info)
    with open(file_path, 'a') as fo:
        fo.write(info + '\n')
        
    return

    
def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
     
     
def adjust_learning_rate(optimizer, epoch, cfg):
    """ Sets the learning rate """
    if cfg.cos:
        lr_min = 0
        lr_max = cfg.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / cfg.num_epochs * 3.1415926535))
    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = cfg.lr * epoch / 5
        elif epoch > 80:
            lr = cfg.lr * 0.01
        elif epoch > 60:
            lr = cfg.lr * 0.1
        else:
            lr = cfg.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, model_dir):
    ''' save ckeck point current and the best '''
    filename = model_dir + '/ckpt/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + '/ckpt/model_best.pth.tar')
     
        
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    
    model.train()
    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)
    end = time.time()
    
    for i, (images, target) in enumerate(tqdm(train_loader)):
        if i > end_steps:
            break

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(cfg.gpu, non_blocking=True)
            target = target.cuda(cfg.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        acc1 = acc1[0].detach().cpu().numpy()
        acc5 = acc5[0].detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        logger('*[Iter]: {:03d} [Acc@1]: {:.3f}%  [Acc@5]: {:.3f}%  [Loss]: {:.5f}.'.format(i, acc1, acc5, loss), flag=False)
        
    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')

    # switch to evaluate mode
    model.eval()

    class_num = torch.zeros(cfg.num_classes).cuda()
    correct = torch.zeros(cfg.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(cfg.gpu, non_blocking=True)


            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1)
            target_one_hot = F.one_hot(target, cfg.num_classes)
            predict_one_hot = F.one_hot(predicted, cfg.num_classes)
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float)
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)

            prob = torch.softmax(output, dim=1)
            confidence_part, pred_class_part = torch.max(prob, dim=1)
            confidence = np.append(confidence, confidence_part.cpu().numpy())
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy())
            true_class = np.append(true_class, target.cpu().numpy())
            
            batch_time.update(time.time() - end)
            end = time.time()

        logger('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%.'.format(top1=top1, top5=top5))

    return top1.avg
