[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Tran2SP

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc).

The original Tran2SP code is distributed under the [MIT License](LICENSE).

The adapted DGL DiffPool code in [`dl/diffpool`](dl/diffpool) is distributed under
the [Apache License 2.0](dl/diffpool/LICENSE). See its
[README](dl/diffpool/README.md) for the source and modifications.

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper
[Tran2SP: Transformer-based Two-Stage Stochastic Programming](https://doi.org/10.1287/ijoc.2025.1226)
by Chanyeong Kim, Hyunglip Bae, Joohwan Ko, Haeun Jeon, and Woo Chang Kim.
Chanyeong Kim and Hyunglip Bae contributed equally as co-first authors.

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

[https://doi.org/10.1287/ijoc.2025.1226](https://doi.org/10.1287/ijoc.2025.1226)

[https://doi.org/10.1287/ijoc.2025.1226.cd](https://doi.org/10.1287/ijoc.2025.1226.cd)

Below is the BibTex for citing this snapshot of the repository.

```bibtex
@misc{Tran2SP2026,
  author =        {Kim, Chanyeong and Bae, Hyunglip and Ko, Joohwan and Jeon, Haeun and Kim, Woo Chang},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Tran2SP: Transformer-based Two-Stage Stochastic Programming}},
  year =          {2026},
  doi =           {10.1287/ijoc.2025.1226.cd},
  url =           {https://github.com/INFORMSJoC/2025.1226},
  note =          {Available for download at https://github.com/INFORMSJoC/2025.1226},
}
```

## Description

The goal of this software is to approximate second-stage value functions in
two-stage stochastic programming using a graph encoder and a Transformer decoder.
The software includes data generation, model training, and solution evaluation
for stochastic multidimensional knapsack (SMKP), stochastic server location (SSLP),
and capacitated facility location (CFLP) problems.

The source code is organized into `bdd/` for optimization models, `dl/` for the
Tran2SP model, `env/` for problem configurations and base data, `scripts/` for
experiment entry points, and `util/` for data processing and evaluation utilities.
The graph encoder in `dl/diffpool/` is adapted from the DiffPool example in
[DGL](https://github.com/dmlc/dgl).

## Building

The software was developed using Python 3.12 with the following package versions.
The PyTorch and DGL builds used here were compiled against CUDA 12.4.

| Package | Version |
| --- | --- |
| PyTorch | 2.4.0 |
| DGL | 2.4.0 |
| Gurobi (gurobipy) | 12.0.2 |
| NumPy | 2.1.3 |
| SciPy | 1.15.2 |
| pandas | 2.2.3 |
| POT | 0.9.5 |
| tqdm | 4.67.1 |
| wandb | 0.20.1 |

A working Gurobi license is required.

To install the dependencies, first install
[PyTorch 2.4.0](https://pytorch.org/get-started/previous-versions/) and a compatible
[DGL build](https://www.dgl.ai/pages/start.html), then execute the following command.

```bash
python -m pip install gurobipy==12.0.2 numpy==2.1.3 scipy==1.15.2 pandas==2.2.3 POT==0.9.5 tqdm==4.67.1 wandb==0.20.1
```

## Replicating

To replicate the data-generation, model-training, and evaluation workflow, execute the following commands from the repository root using Bash. Be sure to generate the training data before the test data, since test normalization uses statistics from the corresponding training dataset. Step 1 uses seed 7 and step 2 uses seed 777.

### 1. Generate training data

To generate the training data, execute the following commands for each problem. Each configuration contains 5,000 samples before outlier filtering, with the scenario count sampled from 1 through `--max_n_scenarios`, inclusive.

| Problem | Instances | Maximum scenarios | Stochastic level<br>(`--sl`) | Samples per configuration |
| --- | --- | --- | --- | --- |
| SMKP | `25`, `50` | 20 | 6 | 5,000 |
| SSLP | `5_25`, `10_50` | 100 | 4 | 5,000 |
| SSLP | `15_45` | 10 | 4 | 5,000 |
| CFLP | `10_10`, `15_15`, `25_25` | 100 | 3 | 5,000 |

**SMKP**

```bash
for instance in 25 50; do
  python -m scripts.run_2SP \
    --prob SMKP --instance "$instance" --max_n_scenarios 20 \
    --test_data False --except_outliers True --sl 6 \
    --save_mode True --num_samples 5000 --seed 7
done
```

**SSLP**

```bash
for instance in 5_25 10_50 15_45; do
  case "$instance" in
    15_45) max_scenarios=10 ;;
    *)     max_scenarios=100 ;;
  esac
  python -m scripts.run_2SP \
    --prob SSLP --instance "$instance" --max_n_scenarios "$max_scenarios" \
    --test_data False --except_outliers True --sl 4 \
    --save_mode True --num_samples 5000 --seed 7
done
```

**CFLP**

```bash
for instance in 10_10 15_15 25_25; do
  python -m scripts.run_2SP \
    --prob CFLP --instance "$instance" --max_n_scenarios 100 \
    --test_data False --except_outliers True --sl 3 \
    --save_mode True --num_samples 5000 --seed 7
done
```

### 2. Generate test data

To generate the test data, execute the following commands. They generate 50 samples for every combination of problem instance, scenario count, and stochastic level listed below, before outlier filtering. Each sample uses exactly `--test_n_scenarios` scenarios.

| Problem | Instances | Test scenarios | Stochastic levels<br>(`--sl`) | Samples per combination |
| --- | --- | --- | --- | --- |
| SMKP | `25`, `50` | 10, 20, 50 | 0–6 | 50 |
| SSLP | `5_25`, `10_50` | 50, 100, 500 | 0–4 | 50 |
| SSLP | `15_45` | 5, 10, 15 | 0–4 | 50 |
| CFLP | `10_10`, `15_15`, `25_25` | 50, 100, 500 | 0–3 | 50 |

**SMKP**

```bash
for instance in 25 50; do
  for n_scenarios in 10 20 50; do
    for sl in 0 1 2 3 4 5 6; do
      python -m scripts.run_2SP \
        --prob SMKP --instance "$instance" --test_n_scenarios "$n_scenarios" \
        --test_data True --except_outliers True --sl "$sl" \
        --save_mode True --num_samples 50 --seed 777
    done
  done
done
```

**SSLP**

```bash
for instance in 5_25 10_50 15_45; do
  case "$instance" in
    15_45) scenarios="5 10 15" ;;
    *)     scenarios="50 100 500" ;;
  esac
  for n_scenarios in $scenarios; do
    for sl in 0 1 2 3 4; do
      python -m scripts.run_2SP \
        --prob SSLP --instance "$instance" --test_n_scenarios "$n_scenarios" \
        --test_data True --except_outliers True --sl "$sl" \
        --save_mode True --num_samples 50 --seed 777
    done
  done
done
```

**CFLP**

```bash
for instance in 10_10 15_15 25_25; do
  for n_scenarios in 50 100 500; do
    for sl in 0 1 2 3; do
      python -m scripts.run_2SP \
        --prob CFLP --instance "$instance" --test_n_scenarios "$n_scenarios" \
        --test_data True --except_outliers True --sl "$sl" \
        --save_mode True --num_samples 50 --seed 777
    done
  done
done
```

Training datasets are saved under `data/<PROB>/<INSTANCE>/train/`. Test datasets are saved under `data/<PROB>/<INSTANCE>/test/sl_<SL>/<N_SCENARIOS>/`. Filtered data and normalized cut labels are stored in each dataset's `clean/` subdirectory. The retained sample count may be smaller than `--num_samples` after filtering. Data generation appends to existing files, so use a fresh copy of the repository for an independent run.

### 3. Train models with random search

To train models for all eight instances, execute the following commands. The settings below use 200 configurations per instance, seed 7, and GPUs 0 and 1. Adjust these values as needed. Set `TRAN2SP_NUM_DEVICE=1` to use only the GPU selected by `TRAN2SP_GPU_START_IDX`. The current multi-GPU implementation supports GPUs 0 and 1.

```bash
TRAN2SP_N_CONFIGS=200
TRAN2SP_SEARCH_SEED=7
TRAN2SP_GPU_START_IDX=0
TRAN2SP_NUM_DEVICE=2

for problem in SMKP SSLP CFLP; do
  case "$problem" in
    SMKP) instances="25 50" ;;
    SSLP) instances="5_25 10_50 15_45" ;;
    CFLP) instances="10_10 15_15 25_25" ;;
  esac
  for instance in $instances; do
    table_file="table_${problem}_${instance}.dat"
    train_file="train_${problem}_${instance}.sh"
    python -m scripts.create_random_search \
      --prob "$problem" --instance "$instance" \
      --n_configs "$TRAN2SP_N_CONFIGS" --seed "$TRAN2SP_SEARCH_SEED" \
      --gpu_start_idx "$TRAN2SP_GPU_START_IDX" --num_device "$TRAN2SP_NUM_DEVICE" \
      --file_name "$table_file"
    cp "$table_file" "$train_file"
    printf '\nwait\n' >> "$train_file"
    bash "$train_file"
  done
done
```

Each instance uses a separate command file and Bash script. The final `wait` lets its training jobs finish before the next instance starts. Checkpoints (`best_val.pt`) and configurations (`config.txt`) are saved under `logs/<PROB>/<INSTANCE>/<RUN_TIMESTAMP>/`.

### 4. Evaluate trained models

After training is complete, execute the following commands to evaluate all saved models on the 123 test combinations from step 2. Change `--device_num` to select the evaluation GPU.

```bash
for problem in SMKP SSLP CFLP; do
  case "$problem" in
    SMKP) instances="25 50"; scenarios="10 20 50"; levels="0 1 2 3 4 5 6" ;;
    SSLP) instances="5_25 10_50 15_45"; scenarios="50 100 500"; levels="0 1 2 3 4" ;;
    CFLP) instances="10_10 15_15 25_25"; scenarios="50 100 500"; levels="0 1 2 3" ;;
  esac
  for instance in $instances; do
    case "$instance" in
      15_45) inst_scenarios="5 10 15" ;;
      *)     inst_scenarios="$scenarios" ;;
    esac
    for n_scenarios in $inst_scenarios; do
      for sl in $levels; do
        python -m scripts.run_inference \
          --prob "$problem" --instance "$instance" \
          --test_n_scenarios "$n_scenarios" --sl "$sl" \
          --except_outliers True --load_model_mode val_loss --device_num 0
      done
    done
  done
done
```

Performance summaries are saved to `logs/<PROB>/<INSTANCE>/results_val_loss_<N_SCENARIOS>_sl<SL>.csv`.

### 5. Select the best models

To select the model with the highest mean `Tran2SP_solution_quality` for each instance, execute the following commands. First complete all evaluations for every candidate. Each SMKP instance has 21 conditions, each SSLP instance has 15, and each CFLP instance has 12. Incomplete runs are not excluded from the ranking.

```bash
for problem in SMKP SSLP CFLP; do
  case "$problem" in
    SMKP) instances="25 50" ;;
    SSLP) instances="5_25 10_50 15_45" ;;
    CFLP) instances="10_10 15_15 25_25" ;;
  esac
  for instance in $instances; do
    python -m scripts.get_best_model \
      --prob "$problem" --instance "$instance" \
      --load_model_mode val_loss --overwrite False
  done
done
```

Each selected run is copied to `logs/<PROB>/<INSTANCE>/best_model/`, with selection details in `source.txt`. Use `--overwrite True` to replace an existing selection.
