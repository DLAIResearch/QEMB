import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data_temps/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--attack_mode", type=str, default="all2one")
    parser.add_argument("--max_order", type=int, default=40)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=500)
    parser.add_argument("--num_workers", type=float, default=8)
    parser.add_argument("--target_label", type=int, default=2)
    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--random_rotation", type=int, default=0)
    parser.add_argument("--random_crop", type=int, default=0)
    return parser
