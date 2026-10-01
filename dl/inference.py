import numpy as np
import torch
import time

from tqdm import tqdm

from util.utils import *
from dl.vf_approximator import *


def inference_instances(args, net, dataloader, test_data_idx):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"

    net = net.to(device)
    net.eval()

    if args.normalized_scenario:
        data_path = join(os.getcwd(), "data", args.prob, args.instance, "test", f"sl_{args.sl}", str(args.test_n_scenarios))
        if args.except_outliers:
            data_path = join(data_path, "clean")
        with open(join(data_path, f"test_scenario_{args.test_n_scenarios}.pickle"), "rb") as f:
            scenario_origin = pickle.load(f)
    else:
        scenario_origin = None

    ef_obj_value_list, tsp_obj_value_list, pred_obj_value_list, error_2sp_list, error_pred_list, t_list  = [], [], [], [], [], []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader)):
            g, y, prob_info, scenario = batch
            instance_pid = test_data_idx[idx]
            
            try:
                scen_origin = scenario_origin[instance_pid]
            except:
                scen_origin = None
            ef_obj_value, tsp_obj_value, pred_obj_value, error_2sp, error_pred, t = inference_one_instance(
                args, net, g, y, scenario, prob_info, scen_origin, num_initial_cut=1
            )
            ef_obj_value_list.append(ef_obj_value)
            tsp_obj_value_list.append(tsp_obj_value)
            pred_obj_value_list.append(pred_obj_value)
            error_2sp_list.append(error_2sp)
            error_pred_list.append(error_pred)
            t_list.append(t)
            
    return (
        np.mean(ef_obj_value_list),
        np.mean(tsp_obj_value_list),
        np.mean(pred_obj_value_list),
        np.mean(error_2sp_list),
        np.std(error_2sp_list),
        np.mean(error_pred_list),
        np.std(error_pred_list),
        np.mean(t_list)
    )


def inference_one_instance(args, net, g, y, scenario, prob_info, scenario_origin, num_initial_cut=1):
    device = f"cuda:{args.device_num}" if torch.cuda.is_available() else "cpu"
    g = g.to(device)
    y = y.to(device)
    scenario = scenario.to(device)
    try:
        inst = prob_info["instance"][0]
    except:
        inst = None

    start = time.time()
    max_length = post_processing(args, save=False)
    pred_y = inference_cuts(net, g, y, scenario, device, max_length, num_initial_cut)

    pred_cuts = pred_y.cpu().numpy()
    pred_end_token = get_end_token_idx(pred_cuts)

    if args.normalized_scenario:
        scenario = scenario_origin
    
    pred_obj_value = calculate_ef_objective_value(args, pred_cuts, pred_end_token, prob_info, scenario, inst)
    t = time.time() - start
    ef_obj_value = prob_info["ef_obj_val"].numpy()[0]
    tsp_obj_value = prob_info["ef_with_2sp_obj_val"].numpy()[0]
    
    if np.isinf(ef_obj_value) | np.isinf(tsp_obj_value):
        if ef_obj_value == tsp_obj_value:
            sq_2sp = np.array([1])
        else:
            sq_2sp = np.array([0])
    else:
        if args.prob == "SSLP":
            sq_2sp = tsp_obj_value / ef_obj_value
        else:
            sq_2sp = ef_obj_value / tsp_obj_value

    if np.isinf(ef_obj_value) | np.isinf(pred_obj_value):
        if ef_obj_value == pred_obj_value:
            sq_pred = np.array([1])
        else:
            sq_pred = np.array([0])
    else:
        if args.prob == "SSLP":
            sq_pred = pred_obj_value / ef_obj_value
        else:
            sq_pred = ef_obj_value / pred_obj_value

    return ef_obj_value, tsp_obj_value, pred_obj_value, sq_2sp, sq_pred, t


def inference_cuts(net, g, y, scenario, device, max_length, num_initial_cut=1):
    pred_y = y[:, :num_initial_cut]
    with torch.no_grad():
        for _ in range(max_length):
            tgt_mask = create_look_ahead_mask(pred_y.size(1)).to(device)
            tgt_mask = tgt_mask == -np.inf

            prediction = net(tgt=pred_y, context_graph=g, scenario=scenario, tgt_mask=tgt_mask)

            logit_to_label = torch.argmax(prediction[:, -1, -4:], dim=1, keepdim=True)
            y_next = torch.unsqueeze(torch.concat((prediction[:, -1, :-4], logit_to_label), dim=1), 1)
            pred_y = torch.concat((pred_y, y_next), dim=1)
    return pred_y
