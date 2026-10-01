import numpy as np
import argparse
import time

import env.sslp.params as sslp_params
import env.cflp.params as cflp_params
import env.smkp.params as smkp_params

from util.utils import *
from env.sslp.dm import *
from env.cflp.dm import *
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

    if args.prob == "SSLP":
        cfg = getattr(sslp_params, f"sslp_{args.instance}")
        cfg.seed = args.seed
        sslp = SSLPDataManager(problem_config=cfg)

        from bdd.sslp import BendersDualDecomposition, MSP

        data_gen_time = []
        while True:
            sslp.generate_instance(args.sl)
            inst = sslp.inst

            bdd = BendersDualDecomposition(
                inst=inst,
                max_n_scenarios=args.max_n_scenarios,
                test_n_scenarios=args.test_n_scenarios,
                test=args.test_data,
            )
            print("Solving 2SP...")
            data_gen_start = time.time()
            bdd.solve()
            data_gen_time.append(time.time() - data_gen_start)
            print("Finish")
            idxs = np.unique(bdd.cut, axis=0, return_index=True)[1]
            cuts = np.array([bdd.cut[idx] for idx in sorted(idxs)])

            info_data = {}
            info_data["gradient"] = cuts[:, :-1]
            info_data["intercept"] = cuts[:, -1:]
            info_data["rv"] = [
                inst["first_stage_costs"],
                inst["second_stage_costs"],
                inst["location_coeffs"],
                inst["client_coeffs"],
            ]
            info_data["scenario"] = bdd.scenario
            info_data["2sp_obj_val"] = bdd.master_obj[-1]
            info_data["2sp_sol"] = bdd.master_sol[-1]
            info_data["instance"] = [inst]

            print("Solving EF with 2sp sol...")
    
            msp = MSP(inst)
            ef_val = msp.evaluate_first_stage_sol(bdd.master_sol[-1], bdd.scenario)

            info_data["ef_with_2sp_obj_val"] = ef_val

            if args.test_data:
                print("Solving EF...")
                msp_model = msp.solve_extensive(scenarios=info_data["scenario"])
                info_data["ef_obj_val"] = msp_model.ObjVal
                print("Finish")

            if args.save_mode:
                data, data_size = save_data(args, info_data, instance=inst)

                print(f"\nSample {data_size}/{args.num_samples} results saved")

                if data_size >= args.num_samples:
                    post_processing(args, save=True, time=np.sum(data_gen_time))
                    normalize_target(args)
                    break

    elif args.prob == "CFLP":
        cfg = getattr(cflp_params, f"cflp_{args.instance}")
        cfg.seed = args.seed
        cflp = FacilityLocationDataManager(problem_config=cfg)

        from bdd.cflp import BendersDualDecomposition, MSP

        data_gen_time = []
        while True:
            cflp.generate_instance(stochastic_level=args.sl)
            inst = cflp.inst

            bdd = BendersDualDecomposition(
                inst=inst,
                max_n_scenarios=args.max_n_scenarios,
                test_n_scenarios=args.test_n_scenarios,
                test=args.test_data,
            )
            print("Solving 2SP...")
            data_gen_start = time.time()
            bdd.solve()
            data_gen_time.append(time.time() - data_gen_start)
            print("Finish")
            idxs = np.unique(bdd.cut, axis=0, return_index=True)[1]
            cuts = np.array([bdd.cut[idx] for idx in sorted(idxs)])

            info_data = {}
            info_data["gradient"] = cuts[:, :-1]
            info_data["intercept"] = cuts[:, -1:]
            info_data["rv"] = [cflp.inst["capacities"], cflp.inst["fixed_costs"], cflp.inst["trans_costs"]]
            info_data["scenario"] = bdd.scenario
            info_data["2sp_obj_val"] = bdd.master_obj[-1]
            info_data["2sp_sol"] = bdd.master_sol[-1]
            info_data["instance"] = [cflp.inst]

            print("Solving EF with 2sp sol...")

            msp = MSP(cflp.inst)
            ef_val = msp.evaluate_first_stage_sol(bdd.master_sol[-1], bdd.scenario)

            info_data["ef_with_2sp_obj_val"] = ef_val

            if args.test_data:
                print("Solving EF...")
                msp_model = msp.solve_extensive(scenarios=info_data["scenario"])
                info_data["ef_obj_val"] = msp_model.ObjVal
                print("Finish")

            if args.save_mode:
                data, data_size = save_data(args, info_data, instance=cflp.inst)

                print(f"\nSample {data_size}/{args.num_samples} results saved")

                if data_size >= args.num_samples:
                    post_processing(args, save=True, time=np.sum(data_gen_time))
                    normalize_target(args)
                    break

    elif args.prob == "SMKP":
        cfg = getattr(smkp_params, f"smkp_{args.instance}")
        cfg.seed = args.seed
        smkp = SMKPDataManager(problem_config=cfg)

        from bdd.smkp import BendersDualDecomposition, MSP

        data_gen_time = []
        while True:
            smkp.generate_instance(stochastic_level=args.sl)
            inst = smkp.inst

            bdd = BendersDualDecomposition(
                inst=inst,
                max_n_scenarios=args.max_n_scenarios,
                test_n_scenarios=args.test_n_scenarios,
                test=args.test_data,
            )
            print("Solving 2SP...")
            data_gen_start = time.time()
            bdd.solve()
            data_gen_time.append(time.time() - data_gen_start)
            print("Finish")
            idxs = np.unique(bdd.cut, axis=0, return_index=True)[1]
            cuts = np.array([bdd.cut[idx] for idx in sorted(idxs)])

            info_data = {}
            info_data["gradient"] = cuts[:, :-1]
            info_data["intercept"] = cuts[:, -1:]
            info_data["rv"] = [inst["c"], inst["d"], inst["A"], inst["C"], inst["W"], inst["T"], inst["b"], inst["h"]]
            info_data["scenario"] = bdd.scenario
            info_data["2sp_obj_val"] = bdd.master_obj[-1]
            info_data["2sp_sol"] = bdd.master_sol[-1]
            info_data["instance"] = [smkp.inst]

            print("Solving EF with 2sp sol...")

            msp = MSP(inst)
            ef_val = msp.evaluate_first_stage_sol(bdd.master_sol[-1], bdd.scenario)

            info_data["ef_with_2sp_obj_val"] = ef_val

            if args.test_data:
                print("Solving EF...")
                msp_model = msp.solve_extensive(scenarios=info_data["scenario"])
                info_data["ef_obj_val"] = msp_model.ObjVal
                print("Finish")

            if args.save_mode:
                data, data_size = save_data(args, info_data, instance=inst)

                print(f"\nSample {data_size}/{args.num_samples} results saved")

                if data_size >= args.num_samples:
                    post_processing(args, save=True, time=np.sum(data_gen_time))
                    normalize_target(args)
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prob", type=str, default="SSLP", choices=["CFLP", "SSLP", "SMKP"])
    parser.add_argument("--instance", type=str, default="5_25")
    parser.add_argument("--max_n_scenarios", type=int, default=100)
    parser.add_argument("--test_n_scenarios", type=int, default=10)

    parser.add_argument("--test_data", type=str2bool, default=True, help="train data or test data")
    parser.add_argument("--except_outliers", type=str2bool, default=True)
    parser.add_argument("--sl", type=int, default=3)
    parser.add_argument("--save_mode", type=str2bool, default=True)
    parser.add_argument("--num_samples", type=int, default=50)

    main(parser.parse_args())
