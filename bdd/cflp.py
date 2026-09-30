import gurobipy as gp
import numpy as np
from multiprocessing import Manager, Pool

QUIET_ENV = gp.Env(empty=True)
QUIET_ENV.setParam("OutputFlag", 0)
QUIET_ENV.start()

PAPER_DEFAULTS = {
    "delta_rule": "paper",
    "kappa": 10,
    "gamma": 1e-1,
    "sub_gap": 5e-3,
    "sub_time": None,
    "stall": 3,
    "term_gap": 1e-2,
    "inout": 0.0,
    "tight": False,
    "tight_rounds": 0,
    "tight_kappa": 0,
    "lam_box": None,
    "max_iter": 2000,
}


CFLP_OVERRIDES = {
    "delta_rule": "adaptive",
    "kappa": 30,
    "gamma": 1e-4,
    "sub_gap": 1e-4,
    "sub_time": 2.0,
    "term_gap": 0.0,
    "inout": 0.5,
    "tight": True,
    "tight_rounds": 3,
    "tight_kappa": 50,
    "lam_box": 2.0,
}


class BendersDualDecomposition:
    def __init__(
        self,
        inst,
        max_n_scenarios,
        test_n_scenarios,
        test,
        params=None,
    ):

        self.p = {**PAPER_DEFAULTS, **CFLP_OVERRIDES, **(params or {})}

        self.inst = inst
        self.n_customers = inst["n_customers"]
        self.n_facilities = inst["n_facilities"]
        self.fixed_cost = np.asarray(inst["fixed_costs"], dtype=float)
        self.trans_cost = np.asarray(inst["trans_costs"], dtype=float)
        self.capacity = np.asarray(inst["capacities"], dtype=float)
        self.integer_second_stage = inst["integer_second_stage"]
        self.recourse_cost = 2 * np.max([np.max(self.fixed_cost), np.max(self.trans_cost)])

        if test:
            self.n_scenarios = test_n_scenarios
        else:
            self.n_scenarios = np.random.randint(1, max_n_scenarios + 1)

        self.scenario = self.get_scenarios(self.n_scenarios)

        self.cut = []
        self.go_st_benders = False
        self.go_lagrangian = False
        self.integer_master = False
        self.check_stop = 0
        self.master_obj = []
        self.master_sol = []
        self.sub_sol = []

        self.inout = self.p["inout"]
        self.core = None
        box = self.p["lam_box"]
        self.lam_box = gp.GRB.INFINITY if box is None else box * self.recourse_cost * self.n_customers
        self.sub_time = gp.GRB.INFINITY if self.p["sub_time"] is None else self.p["sub_time"]
        self.pool = [{} for _ in range(self.n_scenarios)]

        self.master = gp.Model(name="master", env=QUIET_ENV)
        self.master.Params.LogToConsole = 0
        self.m_var = {}

        self.sub = gp.Model(name="sub", env=QUIET_ENV)
        self.sub.Params.LogToConsole = 0
        self.s_var = {}

        self.lag = gp.Model(name="lag", env=QUIET_ENV)
        self.lag.Params.LogToConsole = 0
        self.lag.Params.MIPGap = self.p["sub_gap"]
        self.lag.Params.TimeLimit = self.sub_time
        self.l_var = {}

        self.eval = gp.Model(name="eval", env=QUIET_ENV)
        self.eval.Params.LogToConsole = 0
        self.eval.Params.MIPGap = 0.0
        self.e_var = {}

        self.define_problem()

    def get_scenarios(self, n_scenarios):
        scenarios = np.random.randint(5, 35 + 1, size=(n_scenarios, self.n_customers))

        return scenarios

    def define_second_stage(self, model, var, vtype, copy_vtype):
        var["y"] = model.addMVar(
            (self.n_facilities, self.n_customers), lb=0.0, ub=1.0, vtype=vtype, obj=self.trans_cost, name="y"
        )
        var["u"] = model.addMVar((self.n_customers,), lb=0.0, ub=1.0, vtype=vtype, obj=self.recourse_cost, name="u")
        var["z"] = model.addMVar((self.n_facilities,), lb=0.0, ub=1.0, vtype=copy_vtype, obj=0.0, name="z")

        for j in range(self.n_customers):
            eq_ = var["u"][j]
            for i in range(self.n_facilities):
                eq_ += var["y"][i, j]
            model.addConstr(eq_ >= 1)

        var["capacity_con"] = []
        for i in range(self.n_facilities):
            eq_ = -self.capacity[i] * var["z"][i]
            for j in range(self.n_customers):
                eq_ += 1 * var["y"][i, j]
            var["capacity_con"].append(model.addConstr(eq_ <= 0, name="capacity_con{}".format(i)))

        for i in range(self.n_facilities):
            for j in range(self.n_customers):
                model.addConstr(-var["z"][i] + var["y"][i, j] <= 0)

        model.update()

    def define_problem(self):
        self.m_var["t"] = self.master.addVar(lb=-np.inf, vtype="C", name="t", obj=1.0)
        self.m_var["x"] = self.master.addMVar(
            (self.n_facilities,), lb=0.0, ub=1.0, vtype="B", obj=self.fixed_cost, name="x"
        )
        self.master.update()

        vtype = "B" if self.integer_second_stage else "C"
        self.define_second_stage(self.sub, self.s_var, "C", "C")
        self.define_second_stage(self.lag, self.l_var, vtype, "B")
        self.define_second_stage(self.eval, self.e_var, vtype, "C")

    def set_scenario(self, model, var, s):
        if getattr(model, "_scenario", None) == s:
            return

        scenario = self.scenario[s]
        for i in range(self.n_facilities):
            con = var["capacity_con"][i]
            for j in range(self.n_customers):
                model.chgCoeff(con, var["y"][i, j], scenario[j])
        model._scenario = s

    def add_cut_to_master(self):
        self.master.addConstr(self.m_var["t"] >= self.m_var["x"] @ self.cut[-1][:-1] + self.cut[-1][-1])
        self.master.update()

    def solve_master(self):
        self.master.optimize()

        fs_sol = []
        for i in range(self.n_facilities):
            fs_sol.append(self.m_var["x"][i].x)
        fs_sol = np.array(fs_sol)

        obj = self.master.objval

        return fs_sol, obj

    def solve_sub_eval(self, s, fs_sol):
        self.set_scenario(self.eval, self.e_var, s)
        self.e_var["z"].lb = fs_sol
        self.e_var["z"].ub = fs_sol

        self.eval.update()
        self.eval.optimize()

        obj = self.eval.objval

        self.eval.reset()

        return obj

    def solve_sub_benders(self, s, fs_sol):
        self.set_scenario(self.sub, self.s_var, s)
        if self.sub.getConstrByName("dual[0]") == None:
            self.sub.addConstr(self.s_var["z"] == fs_sol, name="dual")
        else:
            for i in range(self.n_facilities):
                self.sub.getConstrByName("dual[{}]".format(i)).rhs = fs_sol[i]

        self.sub.update()
        self.sub.optimize()

        dual_con = [self.sub.getConstrByName("dual[{}]".format(i)) for i in range(self.n_facilities)]
        lam = np.array([constr.pi for constr in dual_con])
        intercept = self.sub.objval - np.dot(lam, fs_sol)

        self.sub.reset()

        return lam, intercept

    def solve_sub_stbenders(self, s, lam, exact=False):
        self.set_scenario(self.lag, self.l_var, s)
        self.l_var["z"].obj = -np.array(lam)
        if exact:
            self.lag.Params.TimeLimit = gp.GRB.INFINITY
            self.lag.Params.MIPGap = 0.0

        self.lag.update()
        self.lag.optimize()

        intercept = self.lag.objbound
        z_sol = np.round(np.array([self.l_var["z"][i].x for i in range(self.n_facilities)]))
        cost = self.lag.objval + float(np.dot(lam, z_sol))
        key = tuple(z_sol.astype(int))
        self.pool[s][key] = min(self.pool[s].get(key, np.inf), cost)

        self.lag.Params.TimeLimit = self.sub_time
        self.lag.Params.MIPGap = self.p["sub_gap"]
        self.lag.reset()

        return intercept

    def solve_sub_lagrangian(self, s, fs_sol, lam, intercept, kappa=None, delta=1e-5):
        kappa = self.p["kappa"] if kappa is None else kappa
        adaptive = self.p["delta_rule"] == "adaptive"
        best_lam, best_intercept = np.array(lam), intercept
        best_value = intercept + float(np.dot(lam, fs_sol))
        center = np.array(lam)

        qp = gp.Model(env=QUIET_ENV)
        qp.Params.LogToConsole = 0
        lam_var = qp.addMVar((self.n_facilities,), lb=-self.lam_box, ub=self.lam_box, name="lam")
        eta = qp.addVar(lb=-np.inf, name="eta")
        in_model = set()

        def add_columns():
            for key, cost in self.pool[s].items():
                if key not in in_model:
                    qp.addConstr(eta + (np.array(key, dtype=float) - fs_sol) @ lam_var <= cost)
                    in_model.add(key)

        add_columns()
        for t in range(kappa):
            d = delta if adaptive else 1e-2 / (t + 1)
            qp.setObjective(eta - 0.5 * d * (lam_var @ lam_var) + d * (center @ lam_var), gp.GRB.MAXIMIZE)
            qp.optimize()
            if qp.status != gp.GRB.OPTIMAL or eta.x - best_value <= self.p["gamma"] * max(1.0, abs(best_value)):
                break

            new_lam = np.array(lam_var.x)
            new_intercept = self.solve_sub_stbenders(s, new_lam)
            add_columns()
            value = new_intercept + float(np.dot(new_lam, fs_sol))
            if value > best_value + 1e-9 * max(1.0, abs(best_value)):
                best_lam, best_intercept, best_value = new_lam, new_intercept, value
                if adaptive:
                    center, delta = new_lam, max(delta / 2, 1e-8)
            elif adaptive:
                delta = min(delta * 4, 1.0)

            if not adaptive:
                center = new_lam
        qp.dispose()

        return best_lam, best_intercept

    def generate_cut(self, fs_sol, sep_sol, tight=False):
        aver_cut = np.zeros(self.n_facilities + 1)
        for s in range(self.n_scenarios):
            lam, intercept = self.solve_sub_benders(s, sep_sol)

            if self.go_st_benders:
                intercept = self.solve_sub_stbenders(s, lam)
            if self.go_lagrangian:
                lam, intercept = self.solve_sub_lagrangian(s, sep_sol, lam, intercept)

            if tight:
                recourse = self.solve_sub_eval(s, fs_sol)
                for _ in range(self.p["tight_rounds"]):
                    if intercept + float(np.dot(lam, fs_sol)) >= recourse - 1e-6 * max(1.0, abs(recourse)):
                        break
                    intercept = self.solve_sub_stbenders(s, lam, exact=True)
                    lam, intercept = self.solve_sub_lagrangian(s, fs_sol, lam, intercept, kappa=self.p["tight_kappa"])

            aver_cut += np.append(lam, intercept)

        return aver_cut / self.n_scenarios

    def evaluate_first_stage(self, fs_sol):
        recourse = [self.solve_sub_eval(s, fs_sol) for s in range(self.n_scenarios)]

        return float(np.dot(self.fixed_cost, np.round(fs_sol)) + np.mean(recourse))

    def solve(self, max_iter=None):
        max_iter = self.p["max_iter"] if max_iter is None else max_iter
        self.m_var["x"].vtype = "C"
        self.master.update()

        fs_sol = np.zeros(self.n_facilities)
        obj = -1000
        self.master_obj.append(obj)
        self.master_sol.append(fs_sol)

        for _ in range(max_iter):
            if self.integer_master:
                sep_sol = fs_sol
            elif self.core is None:
                sep_sol = self.core = fs_sol
            else:
                sep_sol = (1 - self.inout) * fs_sol + self.inout * self.core
                self.core = 0.5 * self.core + 0.5 * fs_sol

            self.cut.append(self.generate_cut(fs_sol, sep_sol, tight=self.integer_master and self.p["tight"]))
            self.add_cut_to_master()

            fs_sol, obj = self.solve_master()
            if self.integer_master:
                fs_sol = np.round(fs_sol)
            self.master_obj.append(obj)
            self.master_sol.append(fs_sol)

            if self.integer_master:
                ub = self.evaluate_first_stage(fs_sol)
                tol = max(self.p["term_gap"] * abs(ub), 1e-6 * max(1.0, abs(obj)))
                if obj >= ub - tol:
                    break
                continue

            if abs(self.master_obj[-1] - self.master_obj[-2]) <= 1e-6 * max(1.0, abs(obj)):
                self.check_stop += 1
                if self.check_stop >= self.p["stall"]:
                    self.check_stop = 0
                    if not self.go_st_benders:
                        self.go_st_benders = True
                    elif not self.go_lagrangian:
                        self.go_lagrangian = True
                    else:
                        self.integer_master = True
                        self.m_var["x"].vtype = "B"
                        self.master.Params.MIPGap = 0.0
                        self.master.update()
                        fs_sol, obj = self.solve_master()
                        fs_sol = np.round(fs_sol)
                        self.master_obj.append(obj)
                        self.master_sol.append(fs_sol)
            else:
                self.check_stop = 0

        self.sub_sol = [self.solve_sub_eval(s, np.round(fs_sol)) for s in range(self.n_scenarios)]


class MSP:
    def __init__(self, inst):
        self.tol = 1e-6
        self.inst = inst

        self.n_customers = self.inst["n_customers"]
        self.n_facilities = self.inst["n_facilities"]
        self.integer_second_stage = self.inst["integer_second_stage"]
        self.bound_tightening_constrs = self.inst["bound_tightening_constrs"]
        self.capacities = self.inst["capacities"]
        self.fixed_costs = self.inst["fixed_costs"]
        self.trans_costs = self.inst["trans_costs"]
        self.demand = 0
        self.recourse_costs = 2 * np.max([np.max(self.fixed_costs), np.max(self.trans_costs)])

    def _make_extensive_model(self, scenarios):
        demands = scenarios
        self.demand = scenarios
        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios

        model = gp.Model(env=QUIET_ENV)
        var_dict = {}

        for i in range(self.n_facilities):
            var_name = f"x_{i}"
            var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.fixed_costs[i], vtype="B", name=var_name)

        for s in range(n_scenarios):
            for i in range(self.n_facilities):
                for j in range(self.n_customers):
                    var_name = f"y_{i}_{j}_{s}"
                    if self.integer_second_stage:
                        var_dict[var_name] = model.addVar(
                            lb=0.0, ub=1.0, obj=self.trans_costs[i, j] * scenario_prob, vtype="B", name=var_name
                        )
                    else:
                        var_dict[var_name] = model.addVar(
                            lb=0.0, ub=1.0, obj=self.trans_costs[i, j] * scenario_prob, vtype="C", name=var_name
                        )

        for s in range(n_scenarios):
            for j in range(self.n_customers):
                var_name = f"z_{j}_{s}"
                if self.integer_second_stage:
                    var_dict[var_name] = model.addVar(
                        lb=0.0, ub=1.0, obj=self.recourse_costs * scenario_prob, vtype="B", name=var_name
                    )
                else:
                    var_dict[var_name] = model.addVar(
                        lb=0.0, ub=1.0, obj=self.recourse_costs * scenario_prob, vtype="C", name=var_name
                    )

        for s in range(n_scenarios):
            for j in range(self.n_customers):
                cons = var_dict[f"z_{j}_{s}"]
                for i in range(self.n_facilities):
                    cons += var_dict[f"y_{i}_{j}_{s}"]
                model.addConstr(cons >= 1, name=f"d_{j}_{s}")

        for s in range(n_scenarios):
            for i in range(self.n_facilities):
                cons = -self.capacities[i] * var_dict[f"x_{i}"]
                for j in range(self.n_customers):
                    cons += demands[s][j] * var_dict[f"y_{i}_{j}_{s}"]
                model.addConstr(cons <= 0, name=f"c_{i}")

        if self.bound_tightening_constrs:
            for s in range(n_scenarios):
                for i in range(self.n_facilities):
                    for j in range(self.n_customers):
                        model.addConstr(-var_dict[f"x_{i}"] + var_dict[f"y_{i}_{j}_{s}"] <= 0, name=f"t_{i}_{j}_{s}")

        model.update()

        return model

    def get_first_stage_extensive_solution(self, model):
        return self.get_first_stage_solution(model)

    def get_scenarios(self, n_scenarios, test_set):
        test_set = int(test_set)
        rng = np.random.RandomState()
        rng.seed(n_scenarios + test_set)
        scenarios = []
        for _ in range(n_scenarios):
            scenarios.append(rng.randint(5, 35 + 1, size=self.n_customers))

        return scenarios

    def solve_extensive(
        self,
        scenarios,
        gap=0.0001,
        time_limit=3600,
        threads=1,
        log_dir=None,
        node_file_start=None,
        node_file_dir=None,
        test_set="0",
    ):

        def callback(model, where):
            if where == gp.GRB.Callback.MIPSOL:
                self.ef_solving_results["time"].append(model.cbGet(gp.GRB.Callback.RUNTIME))
                self.ef_solving_results["primal"].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST))
                self.ef_solving_results["dual"].append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
                self.ef_solving_results["incumbent"].append(model.cbGetSolution(model._x))

        model = self._make_extensive_model(scenarios)

        model.update()
        self.ef_solving_results = {"primal": [], "dual": [], "incumbent": [], "time": []}
        ef_fs_vars = []
        for i in range(self.n_facilities):
            ef_fs_vars.append(model.getVarByName(f"x_{i}"))
        model._x = ef_fs_vars

        if log_dir is not None:
            model.setParam("LogFile", log_dir)
        if node_file_start is not None:
            model.setParam("NodefileStart", node_file_start)
            model.setParam("NodefileDir", node_file_dir)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)

        model.optimize(callback)
        return model

    def _make_second_stage_model(self, demands):
        model = gp.Model(env=QUIET_ENV)
        var_dict = {}

        for i in range(self.n_facilities):
            var_name = f"x_{i}"
            var_dict[var_name] = model.addVar(obj=self.fixed_costs[i], vtype="B", name=var_name)

        for i in range(self.n_facilities):
            for j in range(self.n_customers):
                var_name = f"y_{i}_{j}"
                if self.integer_second_stage:
                    var_dict[var_name] = model.addVar(obj=self.trans_costs[i, j], vtype="B", name=var_name)
                else:
                    var_dict[var_name] = model.addVar(
                        lb=0.0, ub=1.0, obj=self.trans_costs[i, j], vtype="C", name=var_name
                    )

        for j in range(self.n_customers):
            var_name = f"z_{j}"
            if self.integer_second_stage:
                var_dict[var_name] = model.addVar(obj=self.recourse_costs, vtype="B", name=var_name)
            else:
                var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.recourse_costs, vtype="C", name=var_name)

        model.update()

        for j in range(self.n_customers):
            cons = var_dict[f"z_{j}"]
            for i in range(self.n_facilities):
                cons += var_dict[f"y_{i}_{j}"]
            model.addConstr(cons >= 1, name=f"d_{j}")

        for i in range(self.n_facilities):
            cons = -self.capacities[i] * var_dict[f"x_{i}"]
            for j in range(self.n_customers):
                cons += demands[j] * var_dict[f"y_{i}_{j}"]
            model.addConstr(cons <= 0, name=f"c_{i}")

        if self.bound_tightening_constrs:
            for i in range(self.n_facilities):
                for j in range(self.n_customers):
                    model.addConstr(-var_dict[f"x_{i}"] + var_dict[f"y_{i}_{j}"] <= 0, name=f"t_{i}_{j}")

        model.update()

        return model

    def get_second_stage_cost(self, model):
        second_stage_obj = 0
        for var in model.getVars():
            if "x" not in var.varName:
                second_stage_obj += var.obj * var.x
        return second_stage_obj

    def fix_first_stage(self, model, sol):
        for i, var in enumerate(model.getVars()):
            if "x" in var.varName:
                var.ub = sol[i]
                var.lb = sol[i]
        model.update()
        return model

    def get_second_stage_objective(self, sol, demands, gap=0.0001, time_limit=1e7, threads=1, verbose=0):

        model = self._make_second_stage_model(demands)

        model = self.fix_first_stage(model, sol)

        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)

        model.optimize()

        second_stage_obj = self.get_second_stage_cost(model)
        return second_stage_obj

    def get_first_stage_solution(self, model):
        sol = {}
        for var in model.getVars():
            if "x" not in var.varName:
                continue
            sol[var.varName] = var.x
        return sol

    def evaluate_first_stage_sol(
        self, sol, scenarios, gap=0.0001, time_limit=3600, threads=1, verbose=0, test_set="0", n_procs=1
    ):
        scenario_prob = 1 / len(scenarios)

        first_stage_obj_val = 0
        for i in range(self.n_facilities):
            first_stage_obj_val += sol[i] * self.fixed_costs[i]

        with Manager() as manager:

            mp_list = manager.list()

            pool = Pool(n_procs)
            for demand in scenarios:
                pool.apply_async(
                    self.mp_get_second_stage_obj, args=(sol, demand, scenario_prob, gap, time_limit, verbose, mp_list)
                )
            pool.close()
            pool.join()

            second_stage_costs = list(mp_list)

        second_stage_obj_val = np.sum(second_stage_costs)

        return first_stage_obj_val + second_stage_obj_val

    def mp_get_second_stage_obj(self, sol, demand, scenario_prob, gap, time_limit, verbose, mp_list):
        second_stage_obj = scenario_prob * self.get_second_stage_objective(
            sol, demand, gap=gap, time_limit=time_limit, verbose=verbose
        )
        mp_list.append(second_stage_obj)
