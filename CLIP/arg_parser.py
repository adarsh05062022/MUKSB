# arg_parser.py — MUKSB CLIP
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="MUKSB — Machine Unlearning via Kalai-Smorodinsky Bargaining (CLIP)"
    )
    ##################################### Dataset #################################################
    parser.add_argument("--data",       type=str, default="/data/oxfordpets")
    parser.add_argument("--dataset",    type=str, default="pets")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--data_dir",   type=str, default="/data/oxfordpets")
    parser.add_argument("--num_classes",type=int, default=37)
    ##################################### Architecture ############################################
    parser.add_argument("--arch", type=str, default="ViT-B/32")
    ##################################### General setting ############################################
    parser.add_argument("--seed",     default=2,   type=int)
    parser.add_argument("--gpu",      type=int,    default=0)
    parser.add_argument("--workers",  type=int,    default=4)
    parser.add_argument("--save_dir", default=None, type=str)
    ##################################### Training setting #################################################
    parser.add_argument("--mode",         default="all",   type=str,
                        help="encoder to fine-tune: text, image, or all")
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--lr",           default=0.1, type=float)
    parser.add_argument("--momentum",     default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--epochs",       default=182,  type=int)
    parser.add_argument("--warmup",       default=0,    type=int)
    parser.add_argument("--print_freq",   default=10,   type=int)
    parser.add_argument("--decreasing_lr", default="91,136")
    parser.add_argument("--no-aug",       action="store_true", default=False)
    parser.add_argument("--no-l1-epochs", default=0, type=int)
    ##################################### Unlearn setting #################################################
    parser.add_argument("--unlearn",        type=str,   default="MUKSB",
                        help="unlearning method (MUKSB, MUNBa, FT, GA, SalUn, masked_nash)")
    parser.add_argument("--unlearn_lr",     default=0.01, type=float)
    parser.add_argument("--unlearn_epochs", default=10,   type=int)
    parser.add_argument("--alpha",          default=0.2,  type=float)
    parser.add_argument("--mask",           type=str,     default=None)
    ##################################### SHs setting #################################################
    parser.add_argument("--sparsity",   type=float, default=0.999)
    parser.add_argument("--lam",        type=float, default=0.1)
    parser.add_argument("--project",    action="store_true", default=False)
    parser.add_argument("--memory_num", type=int,   default=10)
    parser.add_argument("--prune_num",  type=int,   default=1)
    parser.add_argument("--shrink",     action="store_true", default=False)
    parser.add_argument("--with_l1",    action="store_true", default=False)
    parser.add_argument("--skip",       action="store_true", default=False)
    ##################################### MUKSB / MUNBa setting #################################################
    parser.add_argument("--beta", type=float, default=1.0,
                        help="scaling factor for forget loss")

    return parser.parse_args()
