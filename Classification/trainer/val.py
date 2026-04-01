"""Classification/trainer/val.py — MUKSB
Validation / evaluation loop.
"""
import os
import sys

import torch

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_MUKSB_CLS = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _MUKSB_CLS not in sys.path:
    sys.path.insert(0, _MUKSB_CLS)

import utils


def validate(val_loader, model, criterion, args):
    """Evaluate model on val_loader; returns top-1 accuracy."""
    losses = utils.AverageMeter()
    top1   = utils.AverageMeter()
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        image  = image.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(image)
            loss   = criterion(output, target)
        prec1 = utils.accuracy(output.float().data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        if i % args.print_freq == 0:
            print(f"Test: [{i}/{len(val_loader)}]\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Accuracy {top1.val:.3f} ({top1.avg:.3f})")

    return top1.avg
