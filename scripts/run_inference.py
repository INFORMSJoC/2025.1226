import numpy as np
import torch
import os
import argparse
import random
import csv
import json
import pandas as pd

import env.cflp.params as cflp_params
import env.sslp.params as sslp_params
import env.smkp.params as smkp_params

from datetime import datetime

from util.utils import *
from dl.dataloader import *
from dl.train import *
from dl.inference import *
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


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_new_parser(args):
    log_path = join(os.getcwd(), "logs", f"{args.prob}", f"{args.instance}", f"{args.run_time}")
    with open(join(log_path, "config.txt")) as f:
        argmuents = json.loads(f.read())

    new_parser = argparse.ArgumentParser(conflict_handler="resolve")
    for arg, val in argmuents.items():
        if arg == "device_num":
            new_parser.add_argument("--device_num", default=args.device_num, type=int)
        elif arg == "test_data":
            new_parser.add_argument("--test_data", default=True, type=str2bool)
        elif arg == "batch_size":
            new_parser.add_argument("--batch_size", default=1, type=int)
        else:
            new_parser.add_argument(f"--{arg}", default=val, type=type(val))
    new_parser.add_argument("--run_time", default=args.run_time, type=str)
    new_parser.add_argument("--load_model_mode", default=args.load_model_mode, type=str)
    new_parser.add_argument("--test_n_scenarios", default=args.test_n_scenarios, type=int)
    new_parser.add_argument("--sl", default=args.sl, type=int)
    new_parser.add_argument("--except_outliers", default=args.except_outliers, type=str2bool)
    args = new_parser.parse_args()
    return args


def main(args):
    set_seed(args.seed)
    device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
    tsp_sq_mean_list = []
    tsp_sq_std_list = []
    pred_sq_mean_list = []
    pred_sq_std_list = []

    ef_obj_values, tsp_obj_values, pred_obj_values = [], [], []

    args = get_new_parser(args)

    num_problem = get_dataset_size(args)
    test_data_idx = np.arange(num_problem)
    test_dataloader = get_2SP_data_loader(args=args, train_idx=None, val_idx=test_data_idx, train=False)

    results = {}

    inst = None
    if args.prob == "CFLP":
        cfg = getattr(cflp_params, f"cflp_{args.instance}")
        cflp = FacilityLocationDataManager(problem_config=cfg)
        cflp.generate_instance(args.sl)
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

    log_path = join(os.getcwd(), "logs", f"{args.prob}", f"{args.instance}", f"{args.run_time}")
    if args.load_model_mode == "val_loss":
        best_file = "best_val.pt"
    elif args.load_model_mode == "error":
        best_file = "best_sol.pt"
    print(f"Load {best_file} Complete")
    net.load_state_dict(torch.load(join(log_path, best_file), map_location=device))

    (
        ef_obj_value,
        tsp_obj_value,
        pred_obj_value,
        tsp_sq_mean,
        tsp_sq_std,
        pred_sq_mean,
        pred_sq_std,
        solving_time,
    ) = inference_instances(args, net, test_dataloader, test_data_idx)

    ef_obj_values.append(ef_obj_value)
    tsp_obj_values.append(tsp_obj_value)
    pred_obj_values.append(pred_obj_value)

    tsp_sq_std_list.append(tsp_sq_std)
    pred_sq_mean_list.append(pred_sq_mean)
    pred_sq_std_list.append(pred_sq_std)

    print(f"2SP solution quality: {tsp_sq_mean} +- {tsp_sq_std}")
    print(f"Predicted solution quality: {pred_sq_mean} +- {pred_sq_std}")
    print(f"EF objective value: {ef_obj_value}")
    print(f"2SP objective value: {tsp_obj_value}")
    print(f"Predicted objective value: {pred_obj_value}")
    print(f"Solving time: {solving_time}\n")

    results["Fold_0"] = {
        "2SP_solution quality": f"{tsp_sq_mean.round(4)} +- {tsp_sq_std.round(4)}",
        "Predicted_solution quality": f"{pred_sq_mean.round(4)} +- {pred_sq_std.round(4)}",
        "EF_objective_value": f"{ef_obj_value.round(4)}",
        "2SP_objective_value": f"{tsp_obj_value.round(4)}",
        "Predicted_objective_value": f"{pred_obj_value.round(4)}",
        "Solving time": f"{solving_time.round(4)}",
    }

    result_file = join(
        os.getcwd(), "logs", f"{args.prob}", f"{args.instance}", f"results_{args.load_model_mode}_{args.test_n_scenarios}_sl{args.sl}.csv"
    )
    with open(result_file, "a+", newline="") as file:
        writer = csv.writer(file)
        if os.path.getsize(result_file) == 0:
            headers = [
                "Run_time",
                "normalized_input",
                "normalized_scenario",
                "assign_dim",
                "diffpool_hidden_dim",
                "d_model",
                "num_decoder_layer",
                "dim_feedforward",
                "Learning_rate",
                "Decaying_rate",
                "Seed",
                "BDD_solution_quality",
                "BDD_solution_quality_std",
                "Tran2SP_solution_quality",
                "Tran2SP_solution_quality_std",
                "EF_obj_val",
                "2SP_obj_val",
                "Predicted_obj_val",
                "training_time",
                "Solving_time",
            ]
            writer.writerow(headers)
        writer.writerow(
            [
                args.run_time,
                args.normalized_feature,
                args.normalized_scenario,
                args.assign_dim,
                args.dp_hidden_dim,
                args.d_model,
                args.num_decoder_layer,
                args.dim_feedforward,
                args.lr,
                args.gamma,
                args.seed,
                tsp_sq_mean.round(4),
                tsp_sq_std.round(4),
                pred_sq_mean.round(4),
                pred_sq_std.round(4),
                ef_obj_value.round(4),
                tsp_obj_value.round(4),
                pred_obj_value.round(4),
                args.training_time,
                solving_time.round(4),
            ]
        )

    return (
        tsp_sq_mean_list,
        tsp_sq_std_list,
        pred_sq_mean_list,
        pred_sq_std_list,
        ef_obj_values,
        tsp_obj_values,
        pred_obj_values,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--prob", type=str, default="SSLP", choices=["CFLP", "SSLP", "SMKP"])
    parser.add_argument("--instance", type=str, default="5_25")
    parser.add_argument("--test_n_scenarios", type=int, default=10)
    parser.add_argument("--sl", type=int, default=6)
    parser.add_argument("--except_outliers", type=str2bool, default=True)
    parser.add_argument("--load_model_mode", type=str, default="val_loss", choices=["val_loss", "error"])
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_time", type=str, default=None)
    log_path = join(os.getcwd(), "logs", f"{parser.parse_args().prob}", f"{parser.parse_args().instance}")
    files = []
    for x in os.listdir(log_path):
        try:
            files.append(datetime.strptime(x, "%Y%m%d_%H%M%S_%f"))
        except ValueError:
            continue
    files.sort()
    files = [datetime.strftime(x, "%Y%m%d_%H%M%S_%f") for x in files]
    for run_time in files:
        args = parser.parse_args()
        args.run_time = run_time

        if os.path.exists(join(log_path, f"results_{args.load_model_mode}_{args.test_n_scenarios}_sl{args.sl}.csv")):
            exist_run_times = pd.read_csv(
                join(log_path, f"results_{args.load_model_mode}_{args.test_n_scenarios}_sl{args.sl}.csv"), usecols=["Run_time"]
            ).values
            if run_time in exist_run_times:
                continue

        (
            tsp_sq_mean,
            tsp_sq_std,
            pred_sq_mean,
            pred_sq_std,
            ef_obj_values,
            tsp_obj_values,
            pred_obj_values,
        ) = main(args)
