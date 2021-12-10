# placeholder for utils
# probably ideally we write some functions to download the data and store it locally
# where it won't be committed to the repo

import torch
import torch.nn as nn
import time
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


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.shape[0]

    _, pred = torch.max(output, dim=-1)

    correct = pred.eq(target).sum() * 1.0

    acc = correct / batch_size

    return acc

def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       1. forward data batch to the model                                  #
        #       2. Compute batch loss                                               #
        #       3. Compute gradients and update model parameters                    #
        #############################################################################
        model.train()
        out = model(data)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        batch_acc = accuracy(out, target)

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses, top1=acc))


def validate(epoch, val_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    num_class = 82
    cm = torch.zeros(num_class, num_class)
    # evaluation loop
    for idx, (data, target) in enumerate(val_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        #############################################################################
        # TODO: Complete the body of training loop                                  #
        #       HINT: torch.no_grad()                                               #
        #############################################################################
        
        model.eval()
        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        batch_acc = accuracy(out, target)

        # update confusion matrix
        _, preds = torch.max(out, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

        losses.update(loss, out.shape[0])
        acc.update(batch_acc, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(val_loader), iter_time=iter_time, loss=losses, top1=acc))
    cm = cm / cm.sum(1)
    per_cls_acc = cm.diag().detach().numpy().tolist()
    for i, acc_i in enumerate(per_cls_acc):
        print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

    print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
    return acc.avg, cm

def adjust_learning_rate(optimizer, epoch, **kwargs):
    warmup = kwargs['warmup']
    learning_rate = kwargs['learning_rate']
    steps=kwargs['steps']

    epoch += 1
    if epoch <= warmup:
        lr = learning_rate * epoch / warmup
    elif epoch > steps[1]:
        lr = learning_rate * 0.01
    elif epoch > steps[0]:
        lr = learning_rate * 0.1
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr