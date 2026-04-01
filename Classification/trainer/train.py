"""Classification/trainer/train.py — MUKSB
Standard training loop with optional mask and L1 regularisation.
"""
import copy
import os
import sys
import time

import torch

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_MUKSB_CLS = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _MUKSB_CLS not in sys.path:
    sys.path.insert(0, _MUKSB_CLS)

import utils


def l1_regularization(model):
    return torch.linalg.norm(torch.cat([p.view(-1) for p in model.parameters()]), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1)
    return optimizer, scheduler


def train(train_loader, model, criterion, optimizer, epoch, args, mask=None, l1=False):
    losses = utils.AverageMeter()
    top1   = utils.AverageMeter()
    model.train()
    start = time.time()

    for i, (image, target) in enumerate(train_loader):
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i + 1, optimizer, len(train_loader), args)
        image  = image.cuda()
        target = target.cuda()
        output = model(image)
        loss   = criterion(output, target)
        if l1:
            loss = loss + args.alpha * l1_regularization(model)
        optimizer.zero_grad()
        loss.backward()
        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
        optimizer.step()

        prec1 = utils.accuracy(output.float().data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                  f"Time {end - start:.2f}")
            start = time.time()

    print(f"train_accuracy {top1.avg:.3f}")
    return top1.avg


def train_with_rewind(model, optimizer, scheduler, train_loader, criterion, args):
    rewind_state_dict = None
    for epoch in range(args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()["param_groups"][0]["lr"])
        train(train_loader, model, criterion, optimizer, epoch, args)
        if (epoch + 1) == args.rewind_epoch:
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, f"epoch_{epoch+1}_rewind_weight.pt"))
            if args.prune_type == "rewind_lt":
                rewind_state_dict = copy.deepcopy(model.state_dict())
        scheduler.step()
        print(f"Epoch duration: {time.time() - start_time:.1f}s")
    return rewind_state_dict
