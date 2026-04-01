"""
Classification/generate_mask.py — MUKSB
Gradient-magnitude saliency mask generation for the forget class.

Usage
-----
  python generate_mask.py --unlearn MUKSB --class_to_replace 1 \\
      --dataset cifar10 --arch resnet18 --gpu 0 \\
      --mask <pretrained.pth> --save_dir masks/cls1

Produces masks/cls1/with_0.1.pt … with_1.0.pt — binary dicts
(param_name → 0/1 tensor) where 1 means "include in gradient update".
"""
import copy
import os
import sys
from collections import OrderedDict

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import arg_parser
import torch
import torch.nn as nn
import unlearn   # MUKSB unlearn package
import utils


def save_gradient_ratio(data_loaders, model, criterion, args):
    """
    Accumulate |∇L_forget| over the forget set, then threshold at multiple
    density levels (0.1 … 1.0) and save binary masks.
    """
    optimizer = torch.optim.SGD(
        model.parameters(), args.unlearn_lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    gradients = {name: 0 for name, _ in model.named_parameters()}
    forget_loader = data_loaders["forget"]
    model.eval()

    for image, target in forget_loader:
        image  = image.cuda()
        target = target.cuda()
        output = model(image)
        loss   = -criterion(output, target)   # ascent direction
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for density in threshold_list:
        all_elements  = -torch.cat([t.flatten() for t in gradients.values()])
        threshold_idx = int(len(all_elements) * density)
        positions = torch.argsort(all_elements)
        ranks     = torch.argsort(positions)
        hard_dict = {}
        start = 0
        for key, tensor in gradients.items():
            n = tensor.numel()
            tensor_ranks = ranks[start: start + n]
            mask = torch.zeros_like(tensor_ranks)
            mask[tensor_ranks < threshold_idx] = 1
            hard_dict[key] = mask.reshape(tensor.shape)
            start += n
        torch.save(hard_dict, os.path.join(args.save_dir, f"with_{density}.pt"))
        print(f"Saved mask with density={density}")


def main():
    args = arg_parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    if args.dataset != "celeba":
        model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
        model.cuda()

        def replace_loader(dataset, shuffle=True):
            utils.setup_seed(seed)
            return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               num_workers=0, pin_memory=True, shuffle=shuffle)

        forget_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data    = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader  = replace_loader(forget_dataset)
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            retain_dataset.data    = retain_dataset.data[retain_dataset.targets >= 0]
            retain_dataset.targets = retain_dataset.targets[retain_dataset.targets >= 0]
            retain_loader  = replace_loader(retain_dataset)
        except Exception:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs    = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader  = replace_loader(forget_dataset)
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            retain_dataset.imgs    = retain_dataset.imgs[retain_dataset.targets >= 0]
            retain_dataset.targets = retain_dataset.targets[retain_dataset.targets >= 0]
            retain_loader  = replace_loader(retain_dataset)
    else:
        model, train_loader_full, val_loader, test_loader, forget_loader, retain_loader = utils.setup_model_dataset(args)
        model.cuda()
        retain_dataset = retain_loader.dataset
        forget_dataset = forget_loader.dataset

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader)
    criterion = nn.CrossEntropyLoss()

    ckpt = torch.load(args.mask, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if args.unlearn != "retrain":
        model.load_state_dict(ckpt, strict=False)

    save_gradient_ratio(unlearn_data_loaders, model, criterion, args)


if __name__ == "__main__":
    main()
