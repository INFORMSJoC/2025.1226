import argparse
import hashlib

import numpy as np


class ContinuousValueSampler(object):

    def __init__(self, lb, ub, prob_zero=0.0):
        self.lb = lb
        self.ub = ub
        self.prob_zero = prob_zero

    def sample(self):
        if np.random.rand() < self.prob_zero:
            return 0
        return np.round(np.random.uniform(self.lb, self.ub), 5)


class DiscreteSampler(object):

    def __init__(self, choices):
        self.choices = choices

    def sample(self):
        return np.random.choice(self.choices)


def get_config():
    config = {
        "normalized_scenario": DiscreteSampler([0]),
        "assign_dim": DiscreteSampler([2, 3, 4, 5, 6, 7, 8]),
        "d_model": DiscreteSampler([8, 16, 32, 64, 128]),
        "dp_hidden_dim": DiscreteSampler([8, 16, 32, 64, 128]),
        "num_decoder_layer": DiscreteSampler([1, 2, 3, 4]),
        "dim_feedforward": DiscreteSampler([16, 32, 64, 128, 256, 512]),
        "scenario_embed_dim1": DiscreteSampler([64, 128, 256, 512]),
        "scenario_embed_dim1": DiscreteSampler([16, 32, 64, 128, 256]),
        "lr": ContinuousValueSampler(1e-4, 5e-3),
        "gamma": ContinuousValueSampler(0.7, 0.99),
        "pos_encoding_max_len": DiscreteSampler([1024]),
        # "pos_encoding_max_len": DiscreteSampler([64])  # For SMKP
    }
    return config


def sample_config(problem, instance, seed, config, i, num_device, gpu_start_idx):
    if num_device > 1:
        config_cmd = (
            f"python -m scripts.train_net --prob {problem} --instance {instance} --seed {seed} --device_num {(i+gpu_start_idx)%2}"
        )
    else:
        config_cmd = (
            f"python -m scripts.train_net --prob {problem} --instance {instance} --seed {seed} --device_num {gpu_start_idx}"
        )
    for param_name, param_sampler in config.items():
        param_val = param_sampler.sample()
        config_cmd += f" --{param_name} {param_val}"

    return config_cmd


def main(args):
    cmds = []

    config = get_config()
    for i in range(args.n_configs):
        p_hash = int(hashlib.md5(b"{ptypes}").hexdigest(), 16)
        np.random.seed((args.seed + i + p_hash) % (2**32 - 1))
        cmds.append(sample_config(args.prob, args.instance, args.seed, config, i, args.num_device, args.gpu_start_idx))

    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmds[:-1]):
        textfile.write(f"{cmd}")
        if (i + 1) % args.num_device == 0:
            textfile.write("\nwait\n")
        else:
            textfile.write(" &\n")
    textfile.write(f"{cmds[-1]}")
    textfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a list of configs to run for random search.")
    parser.add_argument("--prob", type=str, default="SSLP")
    parser.add_argument("--instance", type=str, default="5_25")
    parser.add_argument("--n_configs", type=int, default=200)
    parser.add_argument("--file_name", type=str, default="table.dat")
    parser.add_argument("--gpu_start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_device", type=int, default=1)

    args = parser.parse_args()

    main(args)
