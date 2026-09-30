import os
import re
import glob
import shutil
import argparse

import pandas as pd

from os.path import join


CONDITION_PATTERN = re.compile(r"^results_.+_(?P<n_scenarios>\d+)_sl(?P<sl>\d+)\.csv$")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_condition(csv_file):
    name = os.path.basename(csv_file)
    m = CONDITION_PATTERN.match(name)
    if m is None:
        return name, (9999, 9999)
    sl, n_scenarios = int(m.group("sl")), int(m.group("n_scenarios"))
    return f"sl{sl}/n{n_scenarios}", (sl, n_scenarios)


def collect_results(log_path, load_model_mode):
    csv_files = sorted(glob.glob(join(log_path, f"results_{load_model_mode}_*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No results_{load_model_mode}_*.csv under {log_path}. "
            "Run scripts/run_inference.py first."
        )

    df_list = []
    order = {}
    for f in csv_files:
        label, key = parse_condition(f)
        order[label] = key
        df = pd.read_csv(f)
        df["Condition"] = label
        df_list.append(df)

    columns = [label for label, _ in sorted(order.items(), key=lambda kv: kv[1])]
    return pd.concat(df_list, ignore_index=True), csv_files, columns


def build_table(all_data, metric, columns):
    if metric not in all_data.columns:
        raise KeyError(f"Column '{metric}' not in the results files. Available: {list(all_data.columns)}")

    table = all_data.pivot_table(index="Run_time", columns="Condition", values=metric, aggfunc="mean")
    table = table.reindex(columns=[c for c in columns if c in table.columns])

    summary = all_data.groupby("Run_time")[metric].agg(["mean", "std", "count"])
    table["mean"] = summary["mean"]
    table["std"] = summary["std"]
    table["count"] = summary["count"].astype(int)

    return table.sort_values("mean", ascending=False)


def copy_best(log_path, run_time, dest_name, overwrite):
    src = join(log_path, str(run_time))
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Run directory not found: {src}")

    dest = join(log_path, dest_name)
    if os.path.exists(dest):
        if not overwrite:
            raise FileExistsError(f"{dest} already exists. Pass --overwrite True to replace it.")
        shutil.rmtree(dest)

    shutil.copytree(src, dest)
    return src, dest


def main(args):
    log_path = join(os.getcwd(), "logs", args.prob, args.instance)
    all_data, csv_files, columns = collect_results(log_path, args.load_model_mode)
    table = build_table(all_data, args.metric, columns)

    n_conditions = len(columns)
    print(f"{len(csv_files)} results file(s), {len(all_data)} row(s), "
          f"{len(table)} run time(s), {n_conditions} condition(s)")
    print(f"ranked by mean {args.metric}\n")
    print(table.head(args.top).to_string(float_format=lambda x: f"{x:.4f}", na_rep="-"))

    incomplete = table[table["count"] < n_conditions]
    if len(incomplete):
        print(f"\nnote: {len(incomplete)} run time(s) were not evaluated on every condition; "
              "their mean is taken over fewer rows and is not directly comparable")

    best_run_time = table.index[0]
    best = table.iloc[0]
    print(f"\nbest: {best_run_time}  ({args.metric} = {best['mean']:.6f})")

    if args.dry_run:
        print("dry run, nothing copied")
        return

    src, dest = copy_best(log_path, best_run_time, args.dest_name, args.overwrite)
    print(f"copied {src} -> {dest}")
    for f in sorted(os.listdir(dest)):
        print(f"    {f}")

    with open(join(dest, "source.txt"), "w") as f:
        f.write(f"run_time: {best_run_time}\n")
        f.write(f"metric: {args.metric}\n")
        f.write(f"mean: {best['mean']}\n")
        f.write(f"std: {best['std']}\n")
        f.write(f"n_rows: {int(best['count'])}\n")
        f.write(f"load_model_mode: {args.load_model_mode}\n")
        f.write("per_condition:\n")
        for c in columns:
            if c in table.columns:
                f.write(f"    {c}: {best[c]}\n")
        f.write("results_files:\n")
        for c in csv_files:
            f.write(f"    {os.path.basename(c)}\n")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", type=str, default="SSLP", choices=["CFLP", "SSLP", "SMKP"])
    parser.add_argument("--instance", type=str, default="5_25")
    parser.add_argument("--load_model_mode", type=str, default="val_loss", choices=["val_loss", "error"])
    parser.add_argument("--metric", type=str, default="Tran2SP_solution_quality")
    parser.add_argument("--dest_name", type=str, default="best_model")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--overwrite", type=str2bool, default=True)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    return parser


if __name__ == "__main__":
    main(get_parser().parse_args())
