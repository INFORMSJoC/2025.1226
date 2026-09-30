import numpy as np
import os
import pickle
import gurobipy as gp
import ot
import torch
import dgl
import time
import random

from scipy import stats
from os.path import *
from itertools import product
from copy import deepcopy
from gurobipy import *

from dl.vf_approximator import *
from dl.scheduler import *


def get_dataset_size(args):
    if args.test_data:
        data_path = join(
            os.getcwd(), "data", args.prob, args.instance, "test", f"sl_{args.sl}", str(args.test_n_scenarios)
        )
    else:
        data_path = join(os.getcwd(), "data", args.prob, args.instance, "train")

    if args.except_outliers:
        data_path = join(data_path, "clean")

    if args.test_data:
        with open(join(data_path, f"test_prob_{args.test_n_scenarios}.pickle"), "rb") as f:
            prob = pickle.load(f)
    else:
        with open(join(data_path, "prob.pickle"), "rb") as f:
            prob = pickle.load(f)

    return len(prob)


def get_network_dim(args, instance=None):
    if args.prob == "CFLP":
        src_dim = 2
        if args.include_theta:
            tgt_dim = instance["n_facilities"] + 3
        else:
            tgt_dim = instance["n_facilities"] + 2
        scenario_dim = instance["n_customers"]

    elif args.prob == "SSLP":
        src_dim = 2
        if args.include_theta:
            tgt_dim = instance["n_locations"] + 3
        else:
            tgt_dim = instance["n_locations"] + 2
        scenario_dim = instance["n_clients"]

    elif args.prob == "SMKP":
        src_dim = 56
        if args.include_theta:
            tgt_dim = instance["n_items"] + 3
        else:
            tgt_dim = instance["n_items"] + 2
        scenario_dim = instance["n_items"]

    return src_dim, tgt_dim, scenario_dim


def initialize_model(args, instance=None):
    src_dim, tgt_dim, scenario_dim = get_network_dim(args, instance)
    device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"

    edge_dim = 2 if args.prob == "SSLP" else 1

    net = Tran2SP(
        src_dim=src_dim,
        tgt_dim=tgt_dim,
        scenario_dim=scenario_dim,
        num_gnn_layers=args.num_gnn_layers,
        n_pooling=args.n_pooling,
        assign_dim=args.assign_dim,
        pool_ratio=args.pool_ratio,
        dp_hidden_dim=args.dp_hidden_dim,
        dp_dropout=args.dp_dropout,
        g_out=args.g_out,
        d_model=args.d_model,
        num_decoder_layer=args.num_decoder_layer,
        dim_feedforward=args.dim_feedforward,
        scenario_embed_dim1=args.scenario_embed_dim1,
        scenario_embed_dim2=args.scenario_embed_dim2,
        pos_encoding_max_len=args.pos_encoding_max_len,
        edge_dim=edge_dim,
        training=args.test_data,
        device=device,
    )

    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for layer in net.children():
        for key in list(layer.state_dict().keys()):
            if "bias" in key:
                torch.nn.init.zeros_(layer.state_dict()[key])

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=args.optimizer_eps)

    if args.lr_scheduler == "None":
        lr_scheduler = None
    elif args.lr_scheduler == "Cosine":
        lr_scheduler = CosineWarmUpScheduler(optimizer=optimizer, warmup=args.warm_up, max_iters=args.max_iters)
    elif args.lr_scheduler == "CosineAnnealing":
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=args.first_cycle_steps,
            cycle_mult=args.cycle_mult,
            max_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            gamma=args.gamma,
        )
    elif args.lr_scheduler == "lambdaLR":

        def lr_lambda(epoch):
            return args.lr_lambda**epoch

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    elif args.lr_scheduler == "stepLR":

        def lr_lambda(epoch):
            return args.lr_lambda**epoch

        lr_scheduler = torch.optim.lr_scheduler.StepLR(step_size=args.decay_step_size, gamma=args.step_gamma)
    elif args.lr_scheduler == "cyclic":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            step_size_up=args.step_size_up,
            gamma=args.cyclic_gamma,
            mode=args.cyclic_mode,
        )
    else:
        raise NotImplementedError
    return net, optimizer, lr_scheduler


def save_data(args, info_data, instance=None):
    if args.prob == "CFLP":
        MIN_C, MAX_C = instance["capacities_bound"]
        MIN_F, MAX_F = instance["fixed_costs_bound"]
        MIN_T, MAX_T = instance["trans_costs_bound"]

        norm_rv1 = (((info_data["rv"][0] - MIN_C) / (MAX_C - MIN_C)) * 2) - 1
        norm_rv2 = (((info_data["rv"][1] - MIN_F) / (MAX_F - MIN_F)) * 2) - 1
        norm_rv3 = (((info_data["rv"][2] - MIN_T) / (MAX_T - MIN_T)) * 2) - 1

        fs_node = np.concatenate((norm_rv1.reshape(-1, 1), norm_rv2.reshape(-1, 1)), axis=1)
        node = np.concatenate((fs_node, np.zeros((instance["n_customers"], 2))))
        edge_f = np.zeros(
            (instance["n_facilities"] + instance["n_customers"], instance["n_facilities"] + instance["n_customers"])
        )
        edge_f[: instance["n_facilities"], instance["n_facilities"] :] = norm_rv3
        edge_f = edge_f + edge_f.T - np.diag(edge_f)

        adj = np.ones(
            (instance["n_facilities"] + instance["n_customers"], instance["n_facilities"] + instance["n_customers"])
        )

    elif args.prob == "SSLP":
        MIN_C_FS, MAX_C_FS = instance["first_stage_costs_bound"]
        MIN_C_SS, MAX_C_SS = instance["second_stage_costs_bound"]
        MIN_C_L, MAX_C_L = instance["location_coeffs_bound"]
        MIN_C_C, MAX_C_C = instance["client_coeffs_bound"]

        norm_rv1 = (((info_data["rv"][0] - MIN_C_FS) / (MAX_C_FS - MIN_C_FS)) * 2) - 1
        norm_rv2 = (((info_data["rv"][1] - MIN_C_SS) / (MAX_C_SS - MIN_C_SS)) * 2) - 1
        norm_rv3 = (((info_data["rv"][2] - MIN_C_L) / (MAX_C_L - MIN_C_L)) * 2) - 1
        norm_rv4 = (((info_data["rv"][3] - MIN_C_C) / (MAX_C_C - MIN_C_C)) * 2) - 1

        fs_node = np.concatenate((norm_rv1.reshape(-1, 1), norm_rv3.reshape(-1, 1)), axis=1)
        node = np.concatenate((fs_node, np.zeros((instance["n_clients"], 2))))
        f1 = np.zeros(
            (instance["n_locations"] + instance["n_clients"], instance["n_locations"] + instance["n_clients"])
        )
        f2 = deepcopy(f1)
        f1[: instance["n_locations"], instance["n_locations"] :] = norm_rv2.T
        f2[: instance["n_locations"], instance["n_locations"] :] = norm_rv4.T
        f1 = f1 + f1.T - np.diag(f1)
        f2 = f2 + f2.T - np.diag(f2)
        edge_f = np.concatenate((f1[:, :, np.newaxis], f2[:, :, np.newaxis]), axis=2)

        adj = np.ones(
            (instance["n_locations"] + instance["n_clients"], instance["n_locations"] + instance["n_clients"])
        )

    elif args.prob == "SMKP":
        MIN_c, MAX_c = instance["c_bound"]
        MIN_d, MAX_d = instance["d_bound"]
        MIN_A, MAX_A = instance["A_bound"]
        MIN_C, MAX_C = instance["C_bound"]
        MIN_W, MAX_W = instance["W_bound"]
        MIN_T, MAX_T = instance["T_bound"]

        norm_rv1 = (((info_data["rv"][0] - MIN_c) / (MAX_c - MIN_c)) * 2) - 1
        norm_rv2 = (((info_data["rv"][1] - MIN_d) / (MAX_d - MIN_d)) * 2) - 1
        norm_rv3 = (((info_data["rv"][2] - MIN_A) / (MAX_A - MIN_A)) * 2) - 1
        norm_rv4 = (((info_data["rv"][3] - MIN_C) / (MAX_C - MIN_C)) * 2) - 1
        norm_rv5 = (((info_data["rv"][4] - MIN_W) / (MAX_W - MIN_W)) * 2) - 1
        norm_rv6 = (((info_data["rv"][5] - MIN_T) / (MAX_T - MIN_T)) * 2) - 1

        fs_node = np.concatenate((norm_rv1.reshape(-1, 1), norm_rv3.T, norm_rv6.T), axis=1)
        node_z = np.concatenate((norm_rv2.reshape(-1, 1), norm_rv4.T, np.zeros((instance["n_items"], 5))), axis=1)
        node_y = np.concatenate(
            (np.zeros((instance["n_items"], 1)), np.zeros((instance["n_items"], 50)), norm_rv5.T), axis=1
        )
        node = np.concatenate((fs_node, node_z, node_y))

        adj = np.ones((3 * instance["n_items"], 3 * instance["n_items"]))
        adj[instance["n_items"] : 2 * instance["n_items"], 2 * instance["n_items"] :] = 0
        adj[2 * instance["n_items"] :, instance["n_items"] : 2 * instance["n_items"]] = 0
        edge_f = adj

    np.fill_diagonal(adj, 0)
    g = dgl.graph(np.nonzero(adj))
    g.ndata["feat"] = torch.tensor(node).float()

    if args.prob == "SMKP":
        g.edata["feat"] = g.adj_external().coalesce().values().unsqueeze(1)
        g.edata["feat"] = g.edata["feat"].type(torch.float32)
    else:
        edge_f = edge_f[~np.eye(edge_f.shape[0], dtype=bool)].reshape(adj.shape[0] ** 2 - adj.shape[0], -1)
        g.edata["feat"] = torch.tensor(edge_f).float()

    gradients, intercept = info_data["gradient"], info_data["intercept"]
    cut_info = np.zeros((gradients.shape[0], fs_node.shape[0] + 1))

    if args.prob == "CFLP":
        initial_cut = np.zeros(fs_node.shape[0] + 1)
        initial_cut[-1] = 10000
        cut_info[:, :-1] = gradients
        cut_info[:, -1:] = -intercept
        cut_info = np.concatenate((initial_cut.reshape(1, -1), cut_info), axis=0)

    elif args.prob == "SSLP":
        initial_cut = np.zeros(fs_node.shape[0] + 1)
        initial_cut[-1] = 10000
        cut_info[:, :-1] = gradients
        cut_info[:, -1:] = -intercept
        cut_info = np.concatenate((initial_cut.reshape(1, -1), cut_info), axis=0)

    elif args.prob == "SMKP":
        initial_cut = np.zeros(fs_node.shape[0] + 1)
        initial_cut[-1] = 10000
        cut_info[:, :-1] = gradients
        cut_info[:, -1:] = -intercept
        cut_info = np.concatenate((initial_cut.reshape(1, -1), cut_info), axis=0)

    token_arr = np.ones(cut_info.shape[0])
    token_arr[0] = 0
    token_arr[-1] = 2
    token_arr = (token_arr + 1).reshape(token_arr.shape[0], -1)

    label = np.concatenate((cut_info, token_arr), axis=1)

    data = {
        "prob": [
            {
                "rv": info_data["rv"],
                "2sp_obj_val": info_data["2sp_obj_val"],
                "2sp_sol": info_data["2sp_sol"],
                "ef_with_2sp_obj_val": info_data["ef_with_2sp_obj_val"],
                "instance": info_data["instance"],
            }
        ],
        "context_graph": g,
        "label": label,
        "scenario": info_data["scenario"],
    }

    if args.test_data:
        data["prob"][0]["ef_obj_val"] = info_data["ef_obj_val"]

    if args.test_data:
        data_path = join(
            os.getcwd(), "data", args.prob, args.instance, "test", f"sl_{args.sl}", str(args.test_n_scenarios)
        )
    else:
        data_path = join(os.getcwd(), "data", args.prob, args.instance, "train")

    os.makedirs(data_path, exist_ok=True)

    for key, value in data.items():
        if args.test_data:
            if exists(join(data_path, "test_" + key + f"_{args.test_n_scenarios}.pickle")):
                with open(join(data_path, "test_" + key + f"_{args.test_n_scenarios}.pickle"), "rb") as f:
                    input_data = pickle.load(f)
                if key == "prob":
                    input_data += value
                else:
                    input_data.append(value)
            else:
                input_data = value if key == "prob" else [value]

            with open(join(data_path, "test_" + key + f"_{args.test_n_scenarios}.pickle"), "wb") as fw:
                pickle.dump(input_data, fw)
        else:
            if exists(join(data_path, key + ".pickle")):
                with open(join(data_path, key + ".pickle"), "rb") as f:
                    input_data = pickle.load(f)
                if key == "prob":
                    input_data += value
                else:
                    input_data.append(value)
            else:
                input_data = value if key == "prob" else [value]

            with open(join(data_path, key + ".pickle"), "wb") as fw:
                pickle.dump(input_data, fw)

    if args.test_data:
        with open(join(data_path, "test_prob" + f"_{args.test_n_scenarios}.pickle"), "rb") as f:
            saved_data = pickle.load(f)
        data_size = len(saved_data)
    else:
        with open(join(data_path, "prob" + ".pickle"), "rb") as f:
            saved_data = pickle.load(f)
        data_size = len(saved_data)

    return data, data_size


def normalize_target(args):
    train_data_path = join(os.getcwd(), "data", args.prob, args.instance, "train", "clean")
    test_data_path = join(
        os.getcwd(), "data", args.prob, args.instance, "test", f"sl_{args.sl}", str(args.test_n_scenarios), "clean"
    )

    if args.test_data:
        with open(join(test_data_path, f"test_label_{args.test_n_scenarios}.pickle"), "rb") as f:
            labels = pickle.load(f)
        with open(join(test_data_path, f"test_prob_{args.test_n_scenarios}.pickle"), "rb") as f:
            probs = pickle.load(f)
        with open(join(train_data_path, "prob.pickle"), "rb") as f:
            train_probs = pickle.load(f)
    else:
        with open(join(train_data_path, "label.pickle"), "rb") as f:
            labels = pickle.load(f)
        with open(join(train_data_path, "prob.pickle"), "rb") as f:
            probs = pickle.load(f)

    if args.test_data:
        max_intercept_val, min_intercept_val, max_gradient_val, min_gradient_val = (
            train_probs[0]["max_intercept_value"],
            train_probs[0]["min_intercept_value"],
            train_probs[0]["max_gradient_value"],
            train_probs[0]["min_gradient_value"],
        )
    else:
        max_intercept_val, min_intercept_val, max_gradient_val, min_gradient_val = -100000, 100000, -100000, 100000

        for label in labels:
            if label[1:, -2].max() >= max_intercept_val:
                max_intercept_val = label[1:, -2].max()
            if label[1:, -2].min() <= min_intercept_val:
                min_intercept_val = label[1:, -2].min()
            if label[1:, :-2].max() >= max_gradient_val:
                max_gradient_val = label[1:, :-2].max()
            if label[1:, :-2].min() <= min_gradient_val:
                min_gradient_val = label[1:, :-2].min()

    norm_labels = []
    for label in labels:
        tmp = deepcopy(label)
        tmp[0][-2] = 1
        tmp[1:, :-2] = ((tmp[1:, :-2] - min_gradient_val) / (max_gradient_val - min_gradient_val) * 2) - 1
        tmp[1:, -2] = ((tmp[1:, -2] - min_intercept_val) / (max_intercept_val - min_intercept_val) * 2) - 1
        norm_labels.append(tmp)

    for prob in probs:
        prob["max_intercept_value"] = max_intercept_val
        prob["min_intercept_value"] = min_intercept_val
        prob["max_gradient_value"] = max_gradient_val
        prob["min_gradient_value"] = min_gradient_val

    if args.test_data:
        with open(join(test_data_path, f"test_label_normalized_{args.test_n_scenarios}.pickle"), "wb") as f:
            pickle.dump(norm_labels, f)
        with open(join(test_data_path, f"test_prob_{args.test_n_scenarios}.pickle"), "wb") as f:
            pickle.dump(probs, f)
    else:
        with open(join(train_data_path, "label_normalized.pickle"), "wb") as f:
            pickle.dump(norm_labels, f)
        with open(join(train_data_path, "prob.pickle"), "wb") as f:
            pickle.dump(probs, f)


def get_cross_entropy_loss_weight(args):
    data_path = join(os.getcwd(), "data", args.prob, args.instance, "train")
    if args.except_outliers:
        with open(join(data_path, "clean", "label.pickle"), "rb") as f:
            label_data = pickle.load(f)
    else:
        with open(join(data_path, "label.pickle"), "rb") as f:
            label_data = pickle.load(f)

    num_cuts = [len(label_data[i]) for i in range(len(label_data))]
    max_num_cuts = np.max(num_cuts)
    mean_num_cuts = np.mean(num_cuts)
    weight_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    for label in label_data:
        for c in weight_dict:
            if c == 0:
                weight_dict[c] += max_num_cuts - len(label[:, -1])
            else:
                weight_dict[c] += np.count_nonzero(label[:, -1] == c)

    return [(mean_num_cuts * len(label_data)) / x for x in weight_dict.values()]


def post_processing(args, confidence_level=0.95, save=False, time=None):
    if args.test_data:
        data_path = join(
            os.getcwd(), "data", args.prob, args.instance, "test", f"sl_{args.sl}", str(args.test_n_scenarios)
        )
    else:
        data_path = join(os.getcwd(), "data", args.prob, args.instance, "train")

    if args.test_data:
        with open(join(data_path, f"test_label_{args.test_n_scenarios}.pickle"), "rb") as f:
            label_data = pickle.load(f)
        with open(join(data_path, f"test_context_graph_{args.test_n_scenarios}.pickle"), "rb") as f:
            g_data = pickle.load(f)
        with open(join(data_path, f"test_prob_{args.test_n_scenarios}.pickle"), "rb") as f:
            prob_data = pickle.load(f)
        with open(join(data_path, f"test_scenario_{args.test_n_scenarios}.pickle"), "rb") as f:
            scenario_data = pickle.load(f)
    else:
        with open(join(data_path, "label.pickle"), "rb") as f:
            label_data = pickle.load(f)
        with open(join(data_path, "context_graph.pickle"), "rb") as f:
            g_data = pickle.load(f)
        with open(join(data_path, "prob.pickle"), "rb") as f:
            prob_data = pickle.load(f)
        with open(join(data_path, "scenario.pickle"), "rb") as f:
            scenario_data = pickle.load(f)

    num_cuts = [len(label_data[i]) for i in range(len(label_data))]
    mean = np.mean(num_cuts)
    std = np.std(num_cuts)
    if std != 0:
        confidence_interval = stats.norm.interval(confidence_level, loc=mean, scale=std)
    else:
        confidence_interval = (mean, mean)
    filtered_idx = [
        idx
        for idx in range(len(label_data))
        if confidence_interval[0] <= label_data[idx].shape[0] <= confidence_interval[1]
    ]
    num_filtered_cuts = [len(label_data[i]) for i in filtered_idx]

    if save:
        os.makedirs(join(data_path, "clean"), exist_ok=True)
        if args.test_data:
            with open(join(data_path, "clean", f"test_label_{args.test_n_scenarios}.pickle"), "wb") as f:
                pickle.dump(np.asarray(label_data, dtype="object")[filtered_idx], f)
            with open(join(data_path, "clean", f"test_context_graph_{args.test_n_scenarios}.pickle"), "wb") as f:
                pickle.dump(np.array(g_data)[filtered_idx], f)
            with open(join(data_path, "clean", f"test_prob_{args.test_n_scenarios}.pickle"), "wb") as f:
                pickle.dump(np.array(prob_data)[filtered_idx], f)
            with open(join(data_path, "clean", f"test_scenario_{args.test_n_scenarios}.pickle"), "wb") as f:
                pickle.dump(np.array(scenario_data)[filtered_idx], f)
        else:
            with open(join(data_path, "clean", f"label.pickle"), "wb") as f:
                pickle.dump(np.asarray(label_data, dtype="object")[filtered_idx], f)
            with open(join(data_path, "clean", f"context_graph.pickle"), "wb") as f:
                pickle.dump(np.array(g_data)[filtered_idx], f)
            with open(join(data_path, "clean", f"prob.pickle"), "wb") as f:
                pickle.dump(np.array(prob_data)[filtered_idx], f)
            with open(join(data_path, "clean", f"scenario.pickle"), "wb") as f:
                pickle.dump(np.asarray(scenario_data, dtype="object")[filtered_idx], f)
        if time is not None:
            with open(join(data_path, "clean", f"time.pickle"), "wb") as f:
                pickle.dump({"data_gen_time": time}, f)

    if args.except_outliers:
        return np.max(num_filtered_cuts)
    else:
        return np.max(num_cuts)


def get_end_token_idx(cuts):
    end_token_idx = []
    for cut in cuts:
        token_idx = cut[:, -1]
        try:
            end_token_idx.append((token_idx == 3).nonzero()[0].item())
        except:
            end_token_idx.append(cut.shape[0] - 1)
    return end_token_idx


def solve_2sp_problem(args, cuts_list, end_token_idx, prob_info, instance=None, scenario_info=None):
    obj_value = []
    fs_sol = []
    for i, cuts in enumerate(cuts_list):
        prob = gp.Model()
        prob.Params.LogToConsole = 0
        prob_var = {}

        if args.prob == "CFLP":
            prob_var["t"] = prob.addVar(lb=-np.inf, vtype="C", name="t", obj=1.0)
            fixed_cost = instance["fixed_costs"].numpy()[i]
            prob_var["x"] = prob.addMVar((instance["n_facilities"].numpy()[i],), vtype="B", obj=fixed_cost, name="x")

            if args.normalized_label:
                max_grad_val, min_grad_val, max_intercept_val, min_intercept_val = (
                    prob_info["max_gradient_value"].item(),
                    prob_info["min_gradient_value"].item(),
                    prob_info["max_intercept_value"].item(),
                    prob_info["min_intercept_value"].item(),
                )

            else:
                alpha = 1
                beta = 1

            for cut in cuts[1 : end_token_idx[i] + 1]:
                gradient = 1 / 2 * (cut[:-2] + 1) * (max_grad_val - min_grad_val) + min_grad_val
                intercept = 1 / 2 * (cut[-2] + 1) * (max_intercept_val - min_intercept_val) + min_intercept_val

                prob.addConstr(prob_var["x"] @ gradient - prob_var["t"] <= intercept)

            prob.update()
            prob.optimize()

            try:
                obj_value.append(prob.ObjVal)
                sol = np.array([prob_var["x"][i].x for i in range(instance["n_facilities"])])
            except:
                if prob.Status == 3:
                    obj_value.append(np.inf)
                    sol = np.array([np.inf] * instance["n_facilities"])

                elif prob.Status == 5:
                    obj_value.append(-np.inf)
                    sol = np.array([np.inf] * instance["n_facilities"])
                else:
                    raise ValueError

            fs_sol.append(sol)

        elif args.prob == "SSLP":
            prob_var["t"] = prob.addVar(lb=-np.inf, vtype="C", name="t", obj=1.0)
            first_stage_cost = prob_info["rv"][0].numpy()[i]
            prob_var["x"] = prob.addMVar((instance["n_locations"],), vtype="B", obj=first_stage_cost, name="x")

            eq_ = 0
            for loc in range(instance["n_locations"]):
                eq_ += prob_var["x"][loc]
            prob.addConstr(eq_ <= instance["location_limit"])

            if args.normalized_label:
                max_grad_val, min_grad_val, max_intercept_val, min_intercept_val = (
                    prob_info["max_gradient_value"].item(),
                    prob_info["min_gradient_value"].item(),
                    prob_info["max_intercept_value"].item(),
                    prob_info["min_intercept_value"].item(),
                )

            else:
                alpha = 1
                beta = 1

            for cut in cuts[1 : end_token_idx[i] + 1]:
                gradient = 1 / 2 * (cut[:-2] + 1) * (max_grad_val - min_grad_val) + min_grad_val
                intercept = 1 / 2 * (cut[-2] + 1) * (max_intercept_val - min_intercept_val) + min_intercept_val

                prob.addConstr(prob_var["x"] @ gradient - prob_var["t"] <= intercept)

            prob.update()
            prob.optimize()

            try:
                obj_value.append(prob.ObjVal)
                sol = np.array([prob_var["x"][i].x for i in range(instance["n_locations"])])
            except:
                if prob.Status == 3:
                    obj_value.append(np.inf)
                    sol = np.array([np.inf] * instance["n_locations"])

                elif prob.Status == 5:
                    obj_value.append(-np.inf)
                    sol = np.array([np.inf] * instance["n_locations"])
                else:
                    raise ValueError

            fs_sol.append(sol)

        elif args.prob == "SMKP":
            prob_var["t"] = prob.addVar(lb=-np.inf, vtype="C", name="t", obj=1.0)
            prob_var["x"] = prob.addMVar(
                (instance["n_items"].numpy()[i],), vtype="B", obj=instance["c"].numpy()[i], name="x"
            )
            prob_var["z"] = prob.addMVar(
                (instance["n_items"].numpy()[i],), vtype="B", obj=instance["d"].numpy()[i], name="z"
            )

            eq_ = instance["A"].numpy()[i] @ prob_var["x"] + instance["C"].numpy()[i] @ prob_var["z"]
            prob.addConstr(eq_ >= instance["b"].numpy()[i])

            if args.normalized_label:
                max_grad_val, min_grad_val, max_intercept_val, min_intercept_val = (
                    prob_info["max_gradient_value"].item(),
                    prob_info["min_gradient_value"].item(),
                    prob_info["max_intercept_value"].item(),
                    prob_info["min_intercept_value"].item(),
                )
            else:
                alpha = 1
                beta = 1

            for cut in cuts[1 : end_token_idx[i] + 1]:
                gradient = 1 / 2 * (cut[:-2] + 1) * (max_grad_val - min_grad_val) + min_grad_val
                intercept = 1 / 2 * (cut[-2] + 1) * (max_intercept_val - min_intercept_val) + min_intercept_val

                prob.addConstr(prob_var["x"] @ gradient - prob_var["t"] <= intercept)

            prob.update()
            prob.optimize()

            try:
                obj_value.append(prob.ObjVal)
                sol = {"x": [], "z": []}
                for i in range(instance["n_items"]):
                    sol["x"].append(prob_var["x"][i].x)
                    sol["z"].append(prob_var["z"][i].x)
                sol["x"] = np.array(sol["x"])
                sol["z"] = np.array(sol["z"])
            except:
                if prob.Status == 3:
                    obj_value.append(np.inf)
                elif prob.Status == 5:
                    obj_value.append(-np.inf)
                else:
                    raise ValueError
                sol = {"x": [], "z": []}
                for i in range(instance["n_items"]):
                    sol["x"].append(np.inf)
                    sol["z"].append(np.inf)
                sol["x"] = np.array(sol["x"])
                sol["z"] = np.array(sol["z"])

            fs_sol.append(sol)

    return np.array(obj_value), np.array(fs_sol)


def calculate_ef_objective_value(args, cuts_list, end_token_idx, prob_info, scenario_info, instance=None):
    tsp_obj_val, tsp_sol = solve_2sp_problem(args, cuts_list, end_token_idx, prob_info, instance, scenario_info)
    obj_val = []
    inst = deepcopy(instance)

    if args.prob == "CFLP":
        from bdd.cflp import MSP

        for i in range(len(tsp_sol)):
            scenario = scenario_info.cpu().numpy()[i]

            fixed_cost = instance["fixed_costs"].numpy()[i]
            capacities = instance["capacities"].numpy()[i]
            trans_cost = instance["trans_costs"].numpy()[i]
            inst["fixed_costs"] = fixed_cost
            inst["capacities"] = capacities
            inst["trans_costs"] = trans_cost
            msp = MSP(inst)

            if np.isinf(tsp_sol[i][0]):
                obj_val.append(tsp_sol[i][0])
            else:
                obj_val.append(msp.evaluate_first_stage_sol(tsp_sol[i], scenario))

    elif args.prob == "SSLP":
        from bdd.sslp import MSP

        for i in range(len(tsp_sol)):
            scenario = scenario_info.cpu().numpy()[i]

            first_stage_costs = prob_info["rv"][0].numpy()[i]
            second_stage_costs = prob_info["rv"][1].numpy()[i]
            location_coeffs = prob_info["rv"][2].numpy()[i]
            client_coeffs = prob_info["rv"][3].numpy()[i]

            inst["first_stage_costs"] = first_stage_costs
            inst["second_stage_costs"] = second_stage_costs
            inst["location_coeffs"] = location_coeffs
            inst["client_coeffs"] = client_coeffs

            inst["n_locations"] = inst["n_locations"].numpy()[i]
            inst["n_clients"] = inst["n_clients"].numpy()[i]
            inst["recourse_costs"] = inst["recourse_costs"].numpy()[i]
            inst["location_limit"] = inst["location_limit"].numpy()[i]
            inst["recourse_coeffs"] = inst["recourse_coeffs"].numpy()[i]

            msp = MSP(inst)

            if np.isinf(tsp_sol[i][0]):
                obj_val.append(tsp_sol[i][0])
            else:
                ef_val = []
                for s in scenario:
                    ef = msp.get_second_stage_objective(tsp_sol[i], s) + (tsp_sol[i] * first_stage_costs).sum()
                    ef_val.append(ef)
                obj_val.append(np.mean(ef_val))

    elif args.prob == "SMKP":
        from bdd.smkp import MSP

        for i in range(len(tsp_sol)):
            scenario = scenario_info.cpu().numpy()[i]

            inst["c"] = instance["c"].numpy()[i]
            inst["d"] = instance["d"].numpy()[i]
            inst["A"] = instance["A"].numpy()[i]
            inst["C"] = instance["C"].numpy()[i]
            inst["W"] = instance["W"].numpy()[i]
            inst["T"] = instance["T"].numpy()[i]
            inst["b"] = instance["b"].numpy()[i]
            inst["h"] = instance["h"].numpy()[i]

            msp = MSP(inst)

            if np.isinf(tsp_sol[i]["x"][0]):
                obj_val.append(tsp_sol[i]["x"][0])
            else:
                obj_val.append(msp.evaluate_first_stage_sol(tsp_sol[i], scenario))

    return np.array(obj_val)


def get_n_scenario(scenarios):
    n_scen = []
    for scen in scenarios:
        for i in range(len(scen)):
            if (scen[i] == torch.zeros(scen.shape[1])).all():
                n_scen.append(i)
                break
            else:
                if i == scen.shape[0] - 1:
                    n_scen.append(i + 1)
    return n_scen


def pdist2(points_x, points_y):
    x_norm = torch.sum(points_x**2, dim=1).unsqueeze(1)
    y_norm = torch.sum(points_y**2, dim=1).unsqueeze(0)
    cross = torch.matmul(points_x, points_y.t())
    dist = x_norm - 2 * cross + y_norm
    return dist


def emd_approx_match(points_x, points_y):
    device = f"cuda:{points_x.device.index}"

    n = points_x.shape[0]
    m = points_y.shape[0]
    factorl = max(n, m) / n
    factorr = max(n, m) / m
    pairwise_dist2 = pdist2(points_x, points_y)
    saturatedl = (torch.ones(n, dtype=points_x.dtype) * factorl).to(device)
    saturatedr = (torch.ones(m, dtype=points_y.dtype) * factorr).to(device)
    match = torch.zeros((n, m), dtype=points_x.dtype).to(device)
    scalars = [-(4.0**j) for j in range(8, -3, -1)]
    scalars[-1] = 0.0
    for level in scalars:
        e_sr = saturatedr.unsqueeze(0)
        log_sr = torch.log(e_sr + 1e-30)
        log_weight = pairwise_dist2 * torch.as_tensor(level).to(device) + log_sr
        weight = torch.nn.functional.softmax(log_weight, dim=-1)
        weight = weight * saturatedl.unsqueeze(1)

        ss = torch.sum(weight, dim=0) + 1e-9
        ss = torch.minimum(saturatedr / ss, torch.tensor(1.0))
        weight = weight * ss.unsqueeze(0)
        s = torch.sum(weight, dim=1)
        ss2 = torch.sum(weight, dim=0)
        saturatedl = torch.maximum(saturatedl - s, torch.tensor(0.0))
        match = match + weight
        saturatedr = torch.maximum(saturatedr - ss2, torch.tensor(0.0))
    return match


def emd_approx(points_x, points_y):
    pairwise_dist2 = pdist2(points_x, points_y)
    match = emd_approx_match(points_x, points_y)
    match_cost = torch.sum(pairwise_dist2 * match)
    return match_cost


def batch_emd_approx(pred_params, target_pieces, pred_end_token=None, target_end_token=None, approx=False):
    emd_val = 0
    for i in range(pred_params.shape[0]):
        if pred_end_token:
            pred_y = pred_params[i][: pred_end_token[i] + 1]
        else:
            pred_y = pred_params[i]

        if target_end_token:
            target_y = target_pieces[i][: target_end_token[i] + 1]
        else:
            target_y = target_pieces[i]

        if approx:
            emd_val += emd_approximation(target_y, pred_y)
        else:
            emd_val += earth_movers_distance(target_y, pred_y)

    return emd_val / pred_params.shape[0]


def earth_movers_distance(A, B):
    device = A.device
    M = torch.cdist(A, B, p=2)

    a = torch.from_numpy(ot.unif(A.shape[0])).to(device)
    b = torch.from_numpy(ot.unif(B.shape[0])).to(device)

    emd_distance = ot.emd2(a, b, M)

    return emd_distance


def emd_approximation(A, B):
    dist_matrix = torch.cdist(A, B, p=2)

    min_dist_A_to_B = dist_matrix.min(dim=1)[0].mean()
    min_dist_B_to_A = dist_matrix.min(dim=0)[0].mean()

    emd_approx = (min_dist_A_to_B + min_dist_B_to_A) / 2.0

    return emd_approx


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
