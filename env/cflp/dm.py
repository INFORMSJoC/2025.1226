import numpy as np


class FacilityLocationDataManager:

    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState(seed=self.cfg.seed + 1000)

    def generate_instance(self, stochastic_level):
        self.inst = {}
        self._get_problem_data(self.cfg, self.inst)
        self._generate_stage_data(self.cfg, self.inst, stochastic_level)

    @staticmethod
    def _get_problem_data(cfg, inst):
        inst["n_customers"] = cfg.n_customers
        inst["n_facilities"] = cfg.n_facilities
        inst["integer_second_stage"] = cfg.flag_integer_second_stage
        inst["bound_tightening_constrs"] = cfg.flag_bound_tightening
        inst["time_limit"] = cfg.time_limit
        inst["verbose"] = cfg.verbose

    def _generate_stage_data(self, cfg, inst, sl):
        n_f = cfg.n_facilities
        n_c = cfg.n_customers

        base_rng = np.random.RandomState(cfg.seed)
        base_c_x = base_rng.rand(n_c)
        base_c_y = base_rng.rand(n_c)
        base_f_x = base_rng.rand(n_f)
        base_f_y = base_rng.rand(n_f)
        base_demands = base_rng.randint(5, 36, size=n_c)
        base_cap_raw = base_rng.randint(78, 83, size=n_f)
        base_fc_mult = base_rng.randint(100, 111, size=n_f)
        base_fc_add = base_rng.randint(91, size=n_f)

        inst["demands"] = base_demands
        inst["total_demand"] = inst["demands"].sum()

        if sl >= 1:
            low = int(np.floor(self.rng.uniform(76, 81)))
            high = int(np.floor(self.rng.uniform(80, 85)))
            inst["capacities"] = self.rng.randint(low, high + 1, size=n_f)
        else:
            inst["capacities"] = base_cap_raw.copy()

        if sl >= 2:
            low_m = int(np.floor(self.rng.uniform(98, 102)))
            high_m = int(np.floor(self.rng.uniform(108, 112)))
            fc_mult = self.rng.randint(low_m, high_m + 1, size=n_f)
            fc_add = self.rng.randint(91, size=n_f)
        else:
            fc_mult = base_fc_mult.copy()
            fc_add = base_fc_add.copy()

        inst["fixed_costs"] = (fc_mult * np.sqrt(inst["capacities"]) + fc_add).astype(int)

        inst["c_x"] = base_c_x.copy()
        inst["c_y"] = base_c_y.copy()
        inst["f_x"] = base_f_x.copy()
        inst["f_y"] = base_f_y.copy()

        inst["trans_costs"] = (
            np.sqrt(
                (inst["c_x"].reshape(-1, 1) - inst["f_x"].reshape(1, -1)) ** 2
                + (inst["c_y"].reshape(-1, 1) - inst["f_y"].reshape(1, -1)) ** 2
            )
            * 10
            * inst["demands"].reshape(-1, 1)
        ).T

        if sl >= 3:
            low_t = self.rng.uniform(-5, 0)
            high_t = self.rng.uniform(1, 5)
            inst["trans_costs"] += self.rng.uniform(low_t, high_t, (n_f, n_c))
            inst["trans_costs"] = np.clip(inst["trans_costs"], 0, None)

        inst["total_capacity"] = inst["capacities"].sum()
        inst["capacities"] = (inst["capacities"] * cfg.ratio * inst["total_demand"] / inst["total_capacity"]).astype(
            int
        )
        inst["total_capacity"] = inst["capacities"].sum()

        inst["fixed_costs_bound"] = np.array(
            [
                int(98 * np.sqrt(76)),
                int(112 * np.sqrt(84)) + 90,
            ]
        )
        inst["trans_costs_bound"] = np.array([0.0, np.sqrt(2) * 10 * 35 + 5])
        cap_max_raw = 84
        cap_min_raw = 76
        demand_max_total = 35 * cfg.n_customers
        total_cap_min = (cfg.n_facilities - 1) * cap_min_raw + cap_max_raw
        cap_bound_max = int(cap_max_raw * cfg.ratio * demand_max_total / total_cap_min) + 20
        inst["capacities_bound"] = np.array([0, cap_bound_max])
