import numpy as np
import torch
import os
import torch.nn as nn
import wandb
import time

from tqdm import tqdm, trange

from dl.vf_approximator import *
from util.utils import *


def train(
    args,
    net,
    optimizer,
    lr_scheduler,
    train_dataloader,
    val_dataloader,
    ce_weight,
    n_epochs,
    fold,
    metric,
    device_num,
    run_time,
    n_early_stopping=5,
    save=True,
):
    log_path = os.path.join(os.getcwd(), "logs", f"{args.prob}", args.instance, run_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    net = net.to(device)
    net.train()

    if args.use_wandb:
        wandb.init(project="Tran2SP", group=args.prob)
        wandb.config.update(args)
        wandb.run.name = run_time + f"_fold{fold}"
        wandb.run.save()
        wandb.watch(net)

    cross_entropy_loss = nn.CrossEntropyLoss(weight=torch.Tensor(ce_weight), reduction="mean")
    mse_loss = nn.MSELoss()

    val_loss_tracker = []
    patience = 0

    start = time.time()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1} Training")
        train_loss = []
        train_cut_loss = []
        train_token_loss = []
        start_time = time.time()

        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc="Train Batch Progress"):
            g, y, prob_info, scenario = batch
            g = g.to(device)
            g.ndata["feat"] = g.ndata["feat"].float()
            g.edata["feat"] = g.edata["feat"].float()

            y = y.float().to(device)
            n_scen = get_n_scenario(scenario)
            scenario = scenario.float().to(device)

            input_y = y[:, :-1]
            true_y = y[:, 1:]

            sequence_length = input_y.size(1)
            tgt_mask = create_look_ahead_mask(sequence_length).to(device)
            tgt_mask = tgt_mask == -np.inf

            tgt_pad_mask = create_padding_mask(input_y)

            pred_y = net(
                tgt=input_y,
                context_graph=g,
                scenario=scenario,
                n_scen=n_scen,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )

            token_label = torch.argmax(pred_y[:, :, -4:], dim=2, keepdim=True)

            pred_end_token = get_end_token_idx(torch.concat((pred_y[:, :, :-4], token_label), dim=2))
            true_end_token = get_end_token_idx(true_y)

            cut_loss = args.mse_loss_coeff * mse_loss(
                pred_y[:, :, :-4], true_y[:, :, :-1]
            ) + args.emd_loss_coeff * batch_emd_approx(
                pred_y[:, :, :-4], true_y[:, :, :-1], pred_end_token, true_end_token
            )

            token_loss = (
                cross_entropy_loss(pred_y[:, :, -4:].cpu().transpose(1, 2), true_y[:, :, -1].cpu().to(torch.long))
                * args.token_loss_coeff
            )
            loss = cut_loss + token_loss

            optimizer.zero_grad()
            cut_loss.backward(retain_graph=True)
            token_loss.backward()

            optimizer.step()
            train_loss.append(loss.item())
            train_cut_loss.append(cut_loss.item())
            train_token_loss.append(token_loss.item())

        train_time = time.time() - start_time
        mean_train_loss = np.mean(train_loss)
        mean_train_cut_loss = np.mean(train_cut_loss)
        mean_train_token_loss = np.mean(train_token_loss)
        mean_val_loss = None

        val_metrics = {}
        if val_dataloader:
            print(f"Epoch {epoch + 1} Validation")
            val_loss = []

            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(val_dataloader), desc="Validation Batch Progress"):
                    g, y, prob_info, scenario = batch
                    g = g.to(device)
                    g.ndata["feat"] = g.ndata["feat"].float()
                    g.edata["feat"] = g.edata["feat"].float()

                    y = y.float().to(device)
                    n_scen = get_n_scenario(scenario)
                    scenario = scenario.float().to(device)

                    input_y = y[:, :-1]
                    true_y = y[:, 1:]

                    sequence_length = input_y.size(1)
                    tgt_mask = create_look_ahead_mask(sequence_length).to(device)
                    tgt_mask = tgt_mask == -np.inf

                    tgt_pad_mask = create_padding_mask(input_y)

                    pred_y = net(
                        tgt=input_y,
                        context_graph=g,
                        scenario=scenario,
                        n_scen=n_scen,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_pad_mask,
                    )

                    token_label = torch.argmax(pred_y[:, :, -4:], dim=2, keepdim=True)

                    pred_end_token = get_end_token_idx(torch.concat((pred_y[:, :, :-4], token_label), dim=2))
                    true_end_token = get_end_token_idx(true_y)

                    cut_loss = args.mse_loss_coeff * mse_loss(
                        pred_y[:, :, :-4], true_y[:, :, :-1]
                    ) + args.emd_loss_coeff * batch_emd_approx(
                        pred_y[:, :, :-4], true_y[:, :, :-1], pred_end_token, true_end_token
                    )

                    token_loss = (
                        cross_entropy_loss(
                            pred_y[:, :, -4:].cpu().transpose(1, 2), true_y[:, :, -1].cpu().to(torch.long)
                        )
                        * args.token_loss_coeff
                    )
                    loss = cut_loss + token_loss
                    val_loss.append(loss.item())

            mean_val_loss = np.mean(val_loss)
            val_loss_tracker.append(mean_val_loss)

        if lr_scheduler:
            lr_scheduler.step()

        if args.use_wandb:
            wandb.log(
                {
                    "Train/total_loss": mean_train_loss,
                    "Train/cut_loss": mean_train_cut_loss,
                    "Train/token_loss": mean_train_token_loss,
                    "Train/learning_rate": optimizer.param_groups[0]["lr"],
                    "Validation/total_loss": mean_val_loss,
                }
            )

        print(
            f"Epoch {epoch + 1}/{n_epochs}\n {int(train_time)}s train_loss: {mean_train_loss:.4f} - val_loss: {mean_val_loss:.4f}"
        )

        for metric, score in val_metrics.items():
            print(f"val_{metric}: {score:.4f}")

        if val_loss_tracker[-1] == np.min(val_loss_tracker):
            if save:
                torch.save(net.state_dict(), os.path.join(log_path, f"best_val.pt"))
            patience = 0
        else:
            patience += 1
            if patience >= n_early_stopping:
                args.training_time = time.time() - start
                break

    if args.use_wandb:
        wandb.finish()

    args.training_time = time.time() - start

    if metric == "error":
        performance_metric = -calculate_metric(args, val_dataloader, net, device)
    elif metric == "loss":
        performance_metric = -np.min(val_loss_tracker)

    return performance_metric


def calculate_metric(args, test_dataloader, net, device):
    metric = []
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), desc="Validation Batch Progress"):
            g, y, prob_info, scenario = batch
            g = g.to(device)
            y = y.to(device)
            n_scen = get_n_scenario(scenario)
            scenario = scenario.to(device)
            if args.normalized_scenario:
                scenario = 1 / 2 * scenario * (35 - 5) + 5

            input_y = y[:, :-1]
            true_y = y[:, 1:]

            sequence_length = input_y.size(1)
            tgt_mask = create_look_ahead_mask(sequence_length).to(device)
            tgt_mask = tgt_mask == -np.inf

            tgt_pad_mask = create_padding_mask(input_y)

            pred_y = net(
                tgt=input_y,
                context_graph=g,
                scenario=scenario,
                n_scen=n_scen,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )
            pred_end_token = get_end_token_idx(pred_y.cpu().numpy())

            logit_to_label = torch.argmax(pred_y[:, :, -4:], dim=2, keepdim=True)
            pred_y = torch.concat((pred_y[:, :, :-4], logit_to_label), dim=2)
            pred_y = torch.concat((y[:, 0].unsqueeze(1), pred_y), dim=1)
            obj_values, _ = calculate_ef_objective_value(
                args, pred_y.cpu().numpy(), pred_end_token, prob_info, scenario, prob_info["instance"][0]
            )
            metric.append(
                np.mean(
                    (obj_values - prob_info["ef_with_2sp_obj_val"].numpy())
                    / np.abs(prob_info["ef_with_2sp_obj_val"].numpy())
                )
            )

    return np.mean(metric)
