# arg_parser.py — MUKSB Classification
# Identical to MUNBa Classification arg_parser with MUKSB method added.
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="MUKSB — Machine Unlearning via Kalai-Smorodinsky Bargaining (Classification)"
    )
    ##################################### Dataset #################################################
    parser.add_argument("--data", type=str, default="/datasets/CIFAR10",
                        help="location of the data corpus")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument("--input_size", type=int, default=32, help="size of input images")
    parser.add_argument("--data_dir", type=str,
                        default="/datasets/TinyImageNet/tiny-imagenet-200",
                        help="dir to tiny-imagenet")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument("--arch", type=str, default="resnet18", help="model architecture")
    parser.add_argument("--imagenet_arch", action="store_true",
                        help="architecture for imagenet size samples")
    parser.add_argument("--train_y_file", type=str, default="./labels/train_ys.pth")
    parser.add_argument("--val_y_file",   type=str, default="./labels/val_ys.pth")
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")
    parser.add_argument("--train_seed", default=1, type=int)
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save_dir", default=None, type=str,
                        help="directory to save trained models")
    parser.add_argument("--mask", type=str, default=None, help="sparse model")
    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--epochs", default=182, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument("--decreasing_lr", default="91,136")
    parser.add_argument("--no-aug", action="store_true", default=False)
    parser.add_argument("--no-l1-epochs", default=0, type=int)
    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp")
    parser.add_argument("--pruning_times", default=1, type=int)
    parser.add_argument("--rate",   default=0.95, type=float)
    parser.add_argument("--rate_f", default=0.95, type=float)
    parser.add_argument("--iteration_number",   default=100, type=int)
    parser.add_argument("--iteration_number_f", default=10,  type=int)
    parser.add_argument("--prune_type", default="rewind_lt", type=str)
    parser.add_argument("--random_prune", action="store_true")
    parser.add_argument("--rewind_epoch", default=0, type=int)
    parser.add_argument("--rewind_pth", default=None, type=str)
    ##################################### Unlearn setting #################################################
    parser.add_argument("--unlearn", type=str, default="MUKSB",
                        help="unlearning method (MUKSB, MUNBa, GA, FT, retrain, ...)")
    parser.add_argument("--unlearn_lr",     default=0.01, type=float)
    parser.add_argument("--unlearn_epochs", default=10,   type=int)
    parser.add_argument("--num_indexes_to_replace", type=int, default=None)
    parser.add_argument("--class_to_replace", type=int, default=1,
                        help="class index to forget")
    parser.add_argument("--indexes_to_replace", type=list, default=None)
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise / L1 scale")
    parser.add_argument("--path",  default=None, type=str,  help="mask matrix path")
    ##################################### Attack setting #################################################
    parser.add_argument("--attack", type=str, default="backdoor")
    parser.add_argument("--trigger_size", type=int, default=4)
    ##################################### SHs setting #################################################
    parser.add_argument("--is_retain",   action="store_true", default=False)
    parser.add_argument("--sparsity",    type=float, default=0.9)
    parser.add_argument("--lam",         type=float, default=0.1)
    parser.add_argument("--project",     action="store_true", default=False)
    parser.add_argument("--memory_num",  type=int,   default=10)
    parser.add_argument("--prune_num",   type=int,   default=1)
    parser.add_argument("--shrink",      action="store_true", default=False)
    parser.add_argument("--sam",         action="store_true", default=False)
    parser.add_argument("--with_l1",     action="store_true", default=False)
    ##################################### MUKSB / MUNBa setting #################################################
    parser.add_argument("--beta", type=float, default=1.0,
                        help="scaling factor for forget loss (kept for MUNBa compatibility)")

    return parser.parse_args()
