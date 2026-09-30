import numpy as np
import os
import pickle
import torch
import dgl
import dgl.data

from os.path import *
from dgl.data import DGLDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class ContextGraphDataset(DGLDataset):
    def __init__(self, args, idx_list):
        super().__init__(name="context_graph")

        if args.test_data:
            data_path = join(os.getcwd(), "data", args.prob, args.instance, "test", f"sl_{args.sl}", str(args.test_n_scenarios))
        else:
            data_path = join(os.getcwd(), "data", args.prob, args.instance, "train")

        if args.except_outliers:
            data_path = join(data_path, "clean")

        if args.test_data:
            if args.normalized_feature:
                with open(join(data_path, f"test_context_graph_{args.test_n_scenarios}.pickle"), "rb") as f:
                    g = pickle.load(f)
            if args.normalized_label:
                with open(join(data_path, f"test_label_normalized_{args.test_n_scenarios}.pickle"), "rb") as f:
                    label_raw = pickle.load(f)
            else:
                with open(join(data_path, f"test_label_{args.test_n_scenarios}.pickle"), "rb") as f:
                    label_raw = pickle.load(f)

            with open(join(data_path, f"test_prob_{args.test_n_scenarios}.pickle"), "rb") as f:
                prob_raw = pickle.load(f)

            if args.normalized_scenario:
                with open(join(data_path, f"test_scenario_normalized_{args.test_n_scenarios}.pickle"), "rb") as f:
                    scenario_raw = pickle.load(f)
            else:
                with open(join(data_path, f"test_scenario_{args.test_n_scenarios}.pickle"), "rb") as f:
                    scenario_raw = pickle.load(f)
        else:
            if args.normalized_feature:
                with open(join(data_path, "context_graph.pickle"), "rb") as f:
                    g = pickle.load(f)
            
            if args.normalized_label:
                with open(join(data_path, "label_normalized.pickle"), "rb") as f:
                    label_raw = pickle.load(f)
            else:
                with open(join(data_path, "label.pickle"), "rb") as f:
                    label_raw = pickle.load(f)

            with open(join(data_path, "prob.pickle"), "rb") as f:
                prob_raw = pickle.load(f)

            if args.normalized_scenario:
                with open(join(data_path, "scenario_normalized.pickle"), "rb") as f:
                    scenario_raw = pickle.load(f)
            else:
                with open(join(data_path, "scenario.pickle"), "rb") as f:
                    scenario_raw = pickle.load(f)
            
        self.graphs = np.array(g)[idx_list]
        self.probs = np.array(prob_raw)[idx_list]

        labels = np.asarray(label_raw, dtype="object")[idx_list]
        scenarios = np.asarray(scenario_raw, dtype="object")[idx_list]

        try:
            self.labels = [torch.FloatTensor(y) for y in labels]
        except:
            self.labels = [torch.FloatTensor(y) for y in np.array(label_raw)[idx_list]]
        self.labels = pad_sequence(self.labels, batch_first=True)

        try:
            self.scenarios = [torch.FloatTensor(x) for x in scenarios]
        except:
            self.scenarios = [torch.FloatTensor(x) for x in np.array(scenario_raw)[idx_list]]
        self.scenarios = pad_sequence(self.scenarios, batch_first=True)

    def __len__(self):
        return self.probs.shape[0]    
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.probs[idx], self.scenarios[idx]
    
    def max_seq_length(self):
        return self.labels.shape[1]


def get_2SP_data_loader(args, train_idx, val_idx, train=True):
    dataset = ContextGraphDataset   
    dataloader = dgl.dataloading.GraphDataLoader

    if train:
        train_datasets = dataset(args=args, idx_list=train_idx)
        train_dataloader = dataloader(train_datasets, batch_size=args.batch_size)

    val_datasets = dataset(args=args, idx_list=val_idx)
    val_dataloader = dataloader(val_datasets, batch_size=args.batch_size)

    if not train:
        return val_dataloader
    else:
        return train_dataloader, val_dataloader
