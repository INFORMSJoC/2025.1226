import numpy as np
import os
import argparse
import json


import env.cflp.params as cflp_params
import env.sslp.params as sslp_params
import env.smkp.params as smkp_params

from datetime import datetime

from util.utils import *
from dl.dataloader import *
from dl.train import *
from env.cflp.dm import *
from env.sslp.dm import *
from env.smkp.dm import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args):
    set_seed(args.seed)
    num_problem = get_dataset_size(args)

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if args.save:
        log_path = os.path.join(os.getcwd(), "logs", args.prob, args.instance, run_time)
        os.makedirs(log_path, exist_ok=True)

    inst = None
    if args.prob == "CFLP":
        cfg = getattr(cflp_params, f"cflp_{args.instance}")
        cflp = FacilityLocationDataManager(problem_config=cfg)
        cflp.generate_instance(stochastic_level=3)
        inst = cflp.inst

    elif args.prob == "SSLP":
        cfg = getattr(sslp_params, f"sslp_{args.instance}")
        sslp = SSLPDataManager(problem_config=cfg)
        sslp.generate_instance(stochastic_level=4)
        inst = sslp.inst

    elif args.prob == "SMKP":
        cfg = getattr(smkp_params, f"smkp_{args.instance}")
        smkp = SMKPDataManager(problem_config=cfg)
        smkp.generate_instance(stochastic_level=6)
        inst = smkp.inst

    net, optimizer, lr_scheduler = initialize_model(args, inst)

    n_train_data = int(num_problem * 0.8)
    train_idx = np.random.choice(num_problem, n_train_data, replace=False)
    val_idx = np.setdiff1d(range(num_problem), train_idx)
    train_dataloader, val_dataloader = get_2SP_data_loader(args=args, train_idx=train_idx, val_idx=val_idx, train=True)
    ce_weight = get_cross_entropy_loss_weight(args)

    print(f"Train model for {args.prob}")

    train(
        args=args,
        net=net,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        ce_weight=ce_weight,
        n_epochs=args.n_epochs,
        fold=0,
        metric="loss",
        device_num=args.device_num,
        run_time=run_time,
        save=args.save,
        n_early_stopping=args.n_early_stopping,
    )

    if args.save:
        with open(join(log_path, "config.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

    return run_time


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", type=str, default="SSLP", choices=["CFLP", "SSLP", "SMKP"])
    parser.add_argument("--instance", default="5_25", type=str)
    parser.add_argument("--test_data", type=str2bool, default=False)
    parser.add_argument("--except_outliers", type=str2bool, default=True)
    parser.add_argument("--normalized_feature", type=str2bool, default=True)
    parser.add_argument("--normalized_label", type=str2bool, default=True)
    parser.add_argument("--normalized_scenario", type=str2bool, default=False)
    parser.add_argument("--include_theta", type=str2bool, default=False)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--n_pooling", type=int, default=1)
    parser.add_argument("--assign_dim", type=int, default=5)
    parser.add_argument("--pool_ratio", type=float, default=0.1)
    parser.add_argument("--g_out", type=str, default="mean")
    parser.add_argument("--dp_hidden_dim", type=int, default=128)
    parser.add_argument("--dp_dropout", type=float, default=0.1)
    parser.add_argument("--scenario_embed_dim1", type=int, default=256)
    parser.add_argument("--scenario_embed_dim2", type=int, default=256)
    parser.add_argument("--num_decoder_layer", type=int, default=1)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--pos_encoding_max_len", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.06059)
    parser.add_argument("--optimizer_eps", type=float, default=1e-8)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="CosineAnnealing",
        choices=["None", "Cosine", "CosineAnnealing", "lambdaLR", "stepLR", "cyclic"],
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--token_loss_coeff", type=float, default=1)
    parser.add_argument("--similarity_loss_coeff", type=float, default=1)
    parser.add_argument("--emd_loss_coeff", type=float, default=1)
    parser.add_argument("--mse_loss_coeff", type=float, default=1)

    parser.add_argument("--warm_up", type=int, default=50)
    parser.add_argument("--max_iters", type=int, default=2000)

    parser.add_argument("--first_cycle_steps", type=int, default=10)
    parser.add_argument("--cycle_mult", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.77354)

    parser.add_argument("--lr_lambda", type=float, default=0.95)

    parser.add_argument("--decay_step_size", type=float, default=10)
    parser.add_argument("--step_gamma", type=float, default=0.6)

    parser.add_argument("--base_lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--step_size_up", type=float, default=10)
    parser.add_argument("--cyclic_gamma", type=float, default=0.5)
    parser.add_argument("--cyclic_mode", type=str, default="triangular", choices=["triangular", "triangular2", "exp_range"])
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--n_early_stopping", type=int, default=5)
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
