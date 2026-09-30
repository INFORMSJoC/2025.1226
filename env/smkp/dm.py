import os
import pickle
import gurobipy
import numpy as np


class SMKPDataManager:
    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState(self.cfg.seed)

    def generate_instance(self, stochastic_level):
        print("Generating instance...")

        self.sl = stochastic_level
        self.inst = {}
        self._get_problem_data(self.cfg, self.inst)
        self._get_stage_data(self.cfg, self.inst)
        self._get_perturbed_coeffs()
        self._get_min_max_value()

    @staticmethod
    def _get_problem_data(cfg, inst):
        inst["n_items"] = cfg.n_items
        inst["seed"] = cfg.seed
        inst["data_path"] = cfg.data_path

    def _create_stage_data(self, cfg):
        data = {}
        data["c"] = self.rng.randint(1, 100 + 1, cfg.n_items)
        data["d"] = self.rng.randint(1, 100 + 1, cfg.n_items)
        data["A"] = self.rng.randint(1, 100 + 1, (50, cfg.n_items))
        data["C"] = self.rng.randint(1, 100 + 1, (50, cfg.n_items))
        data["W"] = self.rng.randint(1, 100 + 1, (5, cfg.n_items))
        data["T"] = self.rng.randint(1, 100 + 1, (5, cfg.n_items))
        data["b"] = (3 / 4 * (data["A"] @ np.ones(cfg.n_items) + data["C"] @ np.ones(cfg.n_items))).astype(int)
        data["h"] = (3 / 4 * (data["T"] @ np.ones(cfg.n_items) + data["W"] @ np.ones(cfg.n_items))).astype(int)

        with open(cfg.data_path, "wb") as f:
            pickle.dump(data, f)

    def _get_stage_data(self, cfg, inst):
        if not os.path.exists(inst["data_path"]):
            self._create_stage_data(cfg)
        with open(inst["data_path"], "rb") as f:
            data = pickle.load(f)
        inst.update(data)

    def _get_perturbed_coeffs(self):
        if self.sl >= 1:
            bounds = self.rng.uniform([-5, 1], [0, 5])
            self.inst["c"] += self.rng.randint(bounds[0], bounds[1], self.inst["c"].shape)
            self.inst["c"] = np.clip(self.inst["c"], 1, 100)
        if self.sl >= 2:
            bounds = self.rng.uniform([-5, 1], [0, 5])
            self.inst["d"] += self.rng.randint(bounds[0], bounds[1], self.inst["d"].shape)
            self.inst["d"] = np.clip(self.inst["d"], 1, 100)
        if self.sl >= 3:
            bounds = self.rng.uniform([-5, 1], [0, 5])
            self.inst["A"] += self.rng.randint(bounds[0], bounds[1], self.inst["A"].shape)
            self.inst["A"] = np.clip(self.inst["A"], 1, 100)
        if self.sl >= 4:
            bounds = self.rng.uniform([-5, 1], [0, 5])
            self.inst["C"] += self.rng.randint(bounds[0], bounds[1], self.inst["C"].shape)
            self.inst["C"] = np.clip(self.inst["C"], 1, 100)
        if self.sl >= 5:
            bounds = self.rng.uniform([-5, 1], [0, 5])
            self.inst["W"] += self.rng.randint(bounds[0], bounds[1], self.inst["W"].shape)
            self.inst["W"] = np.clip(self.inst["W"], 1, 100)
        if self.sl >= 6:
            bounds = self.rng.uniform([-5, 1], [0, 5])
            self.inst["T"] += self.rng.randint(bounds[0], bounds[1], self.inst["T"].shape)
            self.inst["T"] = np.clip(self.inst["T"], 1, 100)

        self.inst["b"] = (3 / 4 * (self.inst["A"] @ np.ones(self.inst["n_items"]) + self.inst["C"] @ np.ones(self.inst["n_items"]))).astype(
            int
        )
        self.inst["h"] = (3 / 4 * (self.inst["T"] @ np.ones(self.inst["n_items"]) + self.inst["W"] @ np.ones(self.inst["n_items"]))).astype(
            int
        )

    def _get_min_max_value(self):
        self.inst["c_bound"] = np.array([1, 100])
        self.inst["d_bound"] = np.array([1, 100])
        self.inst["A_bound"] = np.array([1, 100])
        self.inst["C_bound"] = np.array([1, 100])
        self.inst["W_bound"] = np.array([1, 100])
        self.inst["T_bound"] = np.array([1, 100])
        self.inst["b_bound"] = np.array([60 * self.cfg.n_items, 90 * self.cfg.n_items])
        self.inst["h_bound"] = np.array([60 * self.cfg.n_items, 90 * self.cfg.n_items])
