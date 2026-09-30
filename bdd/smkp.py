import gurobipy as gp
import numpy as np
from multiprocessing import Manager, Pool



QUIET_ENV = gp.Env(empty=True)
QUIET_ENV.setParam("OutputFlag", 0)
QUIET_ENV.start()

class BendersDualDecomposition:
    def __init__(self, inst, max_n_scenarios, test_n_scenarios, test):
        self.inst = inst
        self.n_items = inst["n_items"]
        self.c = inst["c"]
        self.d = inst["d"]
        self.A = inst["A"]
        self.C = inst["C"]
        self.b = inst["b"]
        self.W = inst["W"]
        self.T = inst["T"]
        self.h = inst["h"]

        if test:
            self.n_scenarios = test_n_scenarios
        else:
            self.n_scenarios = np.random.randint(1, max_n_scenarios + 1)

        self.scenario = self.get_scenarios(self.n_scenarios)

        self.cut = []
        self.go_st_benders = False
        self.check_stop = 0
        self.master_obj = []
        self.master_sol = []
        self.sub_sol = []

        self.master = gp.Model(name="master", env=QUIET_ENV)
        self.master.Params.LogToConsole = 0
        self.m_var = {}

        self.sub = gp.Model(name="sub", env=QUIET_ENV)
        self.sub.Params.LogToConsole = 0
        self.s_var = {}

        self.define_problem()

    def get_scenarios(self, n_scenarios):
        scenarios = np.random.randint(1, 100 + 1, size=(n_scenarios, self.n_items))
        return scenarios

    def define_problem(self):
        self.m_var["t"] = self.master.addVar(lb=-np.inf, vtype="C", name="t", obj=1.0)
        self.m_var["x"] = self.master.addMVar((self.n_items,), vtype="B", obj=self.c, name="x")
        self.m_var["z"] = self.master.addMVar((self.n_items,), vtype="B", obj=self.d, name="z")

        eq_ = self.A @ self.m_var["x"] + self.C @ self.m_var["z"]
        self.master.addConstr(eq_ >= self.b)
        self.master.update()

        self.s_var["y"] = self.sub.addMVar((self.n_items,), vtype="C", obj=0.0, name="y", ub=1.0, lb=0.0)
        self.s_var["r"] = self.sub.addMVar((self.n_items,), vtype="C", obj=0.0, name="r", ub=1.0, lb=0.0)

        eq_ = self.W @ self.s_var["y"] + self.T @ self.s_var["r"]
        self.sub.addConstr(eq_ >= self.h)
        self.sub.update()

    def add_cut_to_master(self):
        self.master.addConstr(self.m_var["t"] >= self.m_var["x"] @ self.cut[-1][:-1] + self.cut[-1][-1])
        self.master.update()

    def solve_master(self):


        self.master.optimize()

        fs_sol = {"x": [], "z": []}
        for i in range(self.n_items):
            fs_sol["x"].append(self.m_var["x"][i].x)
            fs_sol["z"].append(self.m_var["z"][i].x)
        fs_sol["x"] = np.array(fs_sol["x"])
        fs_sol["z"] = np.array(fs_sol["z"])

        obj = self.master.objval

        self.master.reset()

        return fs_sol, obj

    def solve_sub_benders(self, fs_sol, scenario):
        if self.sub.getConstrByName("dual[0]") == None:
            self.sub.addConstr(self.s_var["r"] == fs_sol["x"], name="dual")
        else:
            for i in range(self.n_items):
                self.sub.getConstrByName(f"dual[{i}]").rhs = fs_sol["x"][i]

        self.s_var["y"].vtype = "C"
        self.s_var["y"].obj = scenario
        self.s_var["r"].vtype = "C"
        self.s_var["r"].obj = 0

        self.sub.update()
        self.sub.optimize()

        dual_con = [x for x in self.sub.getConstrs() if "dual" in x.ConstrName]
        lam = [constr.pi for constr in dual_con]
        intercept = self.sub.objval - np.dot(lam, fs_sol["x"])

        self.sub.reset()

        return lam, intercept

    def solve_sub_stbenders(self, lam):
        for i in range(self.n_items):
            self.sub.remove(self.sub.getConstrByName(f"dual[{i}]"))

        self.s_var["y"].vtype = "B"
        self.s_var["r"].vtype = "B"
        self.s_var["r"].obj = -np.array(lam)

        self.sub.update()
        self.sub.optimize()
        intercept = self.sub.objval

        self.sub.reset()

        return intercept

    def solve(self):
        fs_sol, obj = self.solve_master()

        self.master_obj.append(obj)
        self.master_sol.append(fs_sol)

        while self.check_stop < 4:
            aver_cut = np.zeros(self.n_items + 1)
            for i in range(self.scenario.shape[0]):
                lam, intercept = self.solve_sub_benders(fs_sol, self.scenario[i])

                if self.go_st_benders:
                    intercept = self.solve_sub_stbenders(lam)

                cut = np.append(lam, intercept)
                aver_cut += cut

            self.cut.append(aver_cut / self.scenario.shape[0])
            self.add_cut_to_master()

            fs_sol, obj = self.solve_master()
            self.master_obj.append(obj)
            self.master_sol.append(fs_sol)


            if self.master_obj[-1] == self.master_obj[-2]:
                self.check_stop += 1
                if self.check_stop >= 2:
                    self.go_st_benders = True

            else:
                self.check_stop = 0


class MSP:
    def __init__(self, inst):
        self.tol = 1e-6
        self.inst = inst

        self.n_items = inst["n_items"]
        self.c = inst["c"]
        self.d = inst["d"]
        self.A = inst["A"]
        self.C = inst["C"]
        self.b = inst["b"]
        self.W = inst["W"]
        self.T = inst["T"]
        self.h = inst["h"]

    def _make_extensive_model(self, scenarios):
        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios

        model = gp.Model(env=QUIET_ENV)
        var_dict = {}

        for i in range(self.n_items):
            var_name = f"x_{i + 1}"
            var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.c[i], vtype="B", name=var_name)

        for i in range(self.n_items):
            var_name = f"z_{i + 1}"
            var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=self.d[i], vtype="B", name=var_name)

        for s in range(n_scenarios):
            for i in range(self.n_items):
                var_name = f"y_{i + 1}_{s}"
                var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=scenarios[s][i] * scenario_prob, vtype="B", name=var_name)

        for j in range(len(self.A)):
            eq_ = 0
            for i in range(self.n_items):
                eq_ += self.A[j][i] * var_dict[f"x_{i + 1}"] + self.C[j][i] * var_dict[f"z_{i + 1}"]
            model.addConstr(eq_ >= self.b[j], name=f"first_stage_constr_{j}")

        for s in range(n_scenarios):
            for j in range(len(self.W)):
                eq_ = 0
                for i in range(self.n_items):
                    eq_ += self.W[j][i] * var_dict[f"y_{i + 1}_{s}"] + self.T[j][i] * var_dict[f"x_{i + 1}"]
                model.addConstr(eq_ >= self.h[j], name=f"second_stage_constr_{j}")

        return model


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
        for i in range(1, self.n_items + 1):
            ef_fs_vars.append(model.getVarByName(f"x_{i}"))
        for i in range(1, self.n_items + 1):
            ef_fs_vars.append(model.getVarByName(f"z_{i}"))
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

    def get_first_stage_solution(self, model):
        sol = {}
        for var in model.getVars():
            if "x" not in var.varName:
                continue
            sol[var.varName] = var.x
        return sol

    def evaluate_first_stage_sol(self, sol, scenarios, gap=0.0001, time_limit=600, threads=1, verbose=0, test_set="0", n_procs=1):

        scenario_prob = 1 / len(scenarios)

        first_stage_obj_val = 0
        for i in range(self.n_items):
            first_stage_obj_val += sol["x"][i] * self.c[i]
        for i in range(self.n_items):
            first_stage_obj_val += sol["z"][i] * self.d[i]

        with Manager() as manager:

            mp_list = manager.list()

            pool = Pool(n_procs)
            for demand in scenarios:
                pool.apply_async(self.mp_get_second_stage_obj, args=(sol, demand, scenario_prob, gap, time_limit, verbose, mp_list))
            pool.close()
            pool.join()

            second_stage_costs = list(mp_list)

        second_stage_obj_val = np.sum(second_stage_costs)

        return first_stage_obj_val + second_stage_obj_val

    def mp_get_second_stage_obj(self, sol, scenario, scenario_prob, gap, time_limit, verbose, mp_list):
        second_stage_obj = scenario_prob * self.get_second_stage_objective(sol, scenario, gap=gap, time_limit=time_limit, verbose=verbose)
        mp_list.append(second_stage_obj)

    def get_second_stage_objective(self, sol, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):

        model = self._make_second_stage_model(scenario)

        model = self.fix_first_stage(model, sol)

        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)

        model.optimize()

        second_stage_obj = self.get_second_stage_cost(model)
        return second_stage_obj

    def _make_second_stage_model(self, scenario):
        model = gp.Model(env=QUIET_ENV)
        var_dict = {}

        for i in range(self.n_items):
            var_name = f"x_{i + 1}"
            var_dict[var_name] = model.addVar(obj=self.c[i], vtype="B", name=var_name)

        for i in range(self.n_items):
            var_name = f"z_{i + 1}"
            var_dict[var_name] = model.addVar(obj=self.d[i], vtype="B", name=var_name)
            
        for i in range(self.n_items):
            var_name = f"y_{i + 1}"
            var_dict[var_name] = model.addVar(lb=0.0, ub=1.0, obj=scenario[i], vtype="B", name=var_name)

        model.update()

        for j in range(len(self.W)):
            eq_ = 0
            for i in range(self.n_items):
                eq_ += self.W[j][i] * var_dict[f"y_{i + 1}"] + self.T[j][i] * var_dict[f"x_{i + 1}"]
            model.addConstr(eq_ >= self.h[j], name=f"second_stage_constr_{j}")

        model.update()

        return model

    def fix_first_stage(self, model, sol):
        for i, var in enumerate(model.getVars()):
            if "x" in var.varName:
                var.ub = sol["x"][i]
                var.lb = sol["x"][i]
        model.update()
        return model
    
    def get_second_stage_cost(self, model):
        second_stage_obj = 0
        for var in model.getVars():
            if "x" not in var.varName:
                second_stage_obj += var.obj * var.x
        return second_stage_obj
