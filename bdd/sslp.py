import gurobipy as gp
import numpy as np
from multiprocessing import Pool, Manager



QUIET_ENV = gp.Env(empty=True)
QUIET_ENV.setParam("OutputFlag", 0)
QUIET_ENV.start()

class BendersDualDecomposition:
    def __init__(
        self,
        inst,
        max_n_scenarios,
        test_n_scenarios,
        test
    ):
        self.inst = inst
        self.n_locations = inst["n_locations"]
        self.n_clients = inst["n_clients"]
        self.first_stage_cost = inst["first_stage_costs"]
        self.second_stage_cost = inst["second_stage_costs"]
        self.recourse_cost = inst["recourse_costs"]
        self.location_limit = inst["location_limit"]
        self.location_coeff = inst["location_coeffs"]
        self.recourse_coeff = inst["recourse_coeffs"]
        self.client_coeff = inst["client_coeffs"]
        
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
        scenarios = np.random.randint(0, 2, size=(n_scenarios, self.inst["n_clients"]))

        return scenarios

    def define_problem(self):
        self.m_var["t"] = self.master.addVar(lb=-np.inf, vtype="C", name="t", obj=1.0)
        self.m_var["x"] = self.master.addMVar((self.n_locations,), vtype="B", obj=self.first_stage_cost, name="x")

        eq_ = 0
        for loc in range(self.n_locations):
            eq_ += self.m_var["x"][loc]

        self.master.addConstr(eq_ <= self.location_limit)
        self.master.update()

        self.s_var["y"] = self.sub.addMVar(
            (self.n_clients, self.n_locations), vtype="C", obj=self.second_stage_cost, name="y"
        )
        self.s_var["r"] = self.sub.addMVar((self.n_locations,), vtype="C", obj=self.recourse_cost, name="r")
        self.s_var["z"] = self.sub.addMVar((self.n_locations,), vtype="C", obj=0.0, name="z")

        for loc in range(self.n_locations):
            eq_ = self.location_coeff[loc] * self.s_var["z"][loc]
            eq_ += self.recourse_coeff[loc] * self.s_var["r"][loc]
            for clnt in range(self.n_clients):
                eq_ += self.client_coeff[clnt, loc] * self.s_var["y"][clnt, loc]
            self.sub.addConstr(eq_ >= 0)

        for clnt in range(self.n_clients):
            eq_ = 0
            for loc in range(self.n_locations):
                eq_ += self.s_var["y"][clnt, loc]
            self.sub.addConstr(eq_ == 0, name="clnt_con{}".format(clnt))

        self.sub.update()

    def add_cut_to_master(self):
        self.master.addConstr(self.m_var["t"] >= self.m_var["x"] @ self.cut[-1][:-1] + self.cut[-1][-1])
        self.master.update()

    def solve_master(self):
        self.master.optimize()

        fs_sol = []
        for i in range(self.n_locations):
            fs_sol.append(self.m_var["x"][i].x)
        fs_sol = np.array(fs_sol)

        obj = self.master.objval

        self.master.reset()

        return fs_sol, obj

    def solve_sub_benders(self, scenario, fs_sol):
        if self.sub.getConstrByName("dual[0]") == None:
            self.sub.addConstr(self.s_var["z"] == fs_sol, name="dual")
        else:
            for loc in range(self.n_locations):
                self.sub.getConstrByName("dual[{}]".format(loc)).rhs = fs_sol[loc]

        for clnt in range(self.n_clients):
            self.sub.getConstrByName("clnt_con{}".format(clnt)).rhs = scenario[clnt]

        self.s_var["y"].vtype = "C"
        self.s_var["z"].vtype = "C"
        self.s_var["z"].obj = 0

        self.sub.update()
        self.sub.optimize()
        
        s_sol = []
        for i in range(self.n_clients):
            for j in range(self.n_locations):
                s_sol.append(self.s_var["y"][i][j].x)
        s_sol = np.array(s_sol)

        dual_con = self.sub.getConstrs()
        dual_con = dual_con[-self.n_locations :]
        lam = [constr.pi for constr in dual_con]
        intercept = self.sub.objval - np.dot(lam, fs_sol)

        self.sub.reset()

        return lam, intercept, s_sol

    def solve_sub_stbenders(self, lam):
        for loc in range(self.n_locations):
            self.sub.remove(self.sub.getConstrByName("dual[{}]".format(loc)))

        self.s_var["y"].vtype = "B"
        self.s_var["z"].vtype = "B"
        self.s_var["z"].obj = -np.array(lam)

        self.sub.update()
        self.sub.optimize()
        
        s_sol = []
        for i in range(self.n_clients):
            for j in range(self.n_locations):
                s_sol.append(self.s_var["y"][i][j].x)
        s_sol = np.array(s_sol)

        intercept = self.sub.objval

        self.sub.reset()

        return intercept, s_sol

    def solve(self):
        fs_sol = np.zeros(self.n_locations)
        obj = -1000
        self.master_obj.append(obj)
        self.master_sol.append(fs_sol)

        while self.check_stop < 4:
            aver_cut = np.zeros(self.n_locations + 1)
            sub_sol = []
            for i in range(self.scenario.shape[0]):
                lam, intercept, s_sol = self.solve_sub_benders(scenario=self.scenario[i, :], fs_sol=fs_sol)

                if self.go_st_benders:
                    intercept, s_sol = self.solve_sub_stbenders(lam)

                cut = np.append(lam, intercept)
                aver_cut += cut
                sub_sol.append(s_sol)

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
        self.sub_sol = sub_sol

class MSP:
    def __init__(self, inst, relative_dir=None):
        self.inst = inst
        self.scenario = 0

    def _make_extensive_model(self, scenarios):

        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios
        self.scenario = scenarios

        model = gp.Model(env=QUIET_ENV)

        x_vars, y_vars, r_vars = {}, {}, {}

        for loc in range(self.inst["n_locations"]):
            v_name = f"x_{loc + 1}"
            obj = self.inst["first_stage_costs"][loc]
            x_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for scen in range(n_scenarios):
            for clnt in range(self.inst["n_clients"]):
                for loc in range(self.inst["n_locations"]):
                    v_name = f"y_{clnt + 1}_{loc + 1}_{scen}"
                    obj = self.inst["second_stage_costs"][clnt][loc] * scenario_prob
                    y_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for scen in range(n_scenarios):
            for loc in range(self.inst["n_locations"]):
                v_name = f"r_{loc + 1}_{scen}"
                obj = self.inst["recourse_costs"][loc] * scenario_prob
                r_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="C")

        eq_ = 0
        for loc in range(self.inst["n_locations"]):
            eq_ += x_vars[f"x_{loc + 1}"]

        model.addConstr(eq_ <= self.inst["location_limit"], name="location_limit")

        for scen in range(n_scenarios):
            for loc in range(self.inst["n_locations"]):
                eq_ = self.inst["location_coeffs"][loc] * x_vars[f"x_{loc + 1}"]
                eq_ += self.inst["recourse_coeffs"][loc] * r_vars[f"r_{loc + 1}_{scen}"]
                for clnt in range(self.inst["n_clients"]):
                    eq_ += self.inst["client_coeffs"][clnt][loc] * y_vars[f"y_{clnt + 1}_{loc + 1}_{scen}"]
                model.addConstr(eq_ >= 0, name=f"capacity_{loc + 1}_{scen}")

        for scen in range(n_scenarios):
            for clnt in range(self.inst["n_clients"]):
                eq_ = 0
                for loc in range(self.inst["n_locations"]):
                    eq_ += y_vars[f"y_{clnt + 1}_{loc + 1}_{scen}"]
                model.addConstr(eq_ == scenarios[scen][clnt], name=f"active_{clnt + 1}_{scen}")

        return model

    def _make_second_stage_model(self, scenario):
        model = gp.Model(env=QUIET_ENV)

        x_vars, y_vars, r_vars = {}, {}, {}

        for loc in range(self.inst["n_locations"]):
            v_name = f"x_{loc + 1}"
            obj = self.inst["first_stage_costs"][loc]
            x_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for clnt in range(self.inst["n_clients"]):
            for loc in range(self.inst["n_locations"]):
                v_name = f"y_{clnt + 1}_{loc + 1}"
                obj = self.inst["second_stage_costs"][clnt][loc]
                y_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="B")

        for loc in range(self.inst["n_locations"]):
            v_name = f"r_{loc + 1}"
            obj = self.inst["recourse_costs"][loc]
            r_vars[v_name] = model.addVar(name=v_name, obj=obj, vtype="C")

        constrs = {}

        eq_ = 0
        for loc in range(self.inst["n_locations"]):
            eq_ += x_vars[f"x_{loc + 1}"]

        c_name = "location_limit"
        constrs[c_name] = model.addConstr(eq_ <= self.inst["location_limit"], name=c_name)

        for loc in range(self.inst["n_locations"]):
            eq_ = self.inst["location_coeffs"][loc] * x_vars[f"x_{loc + 1}"]
            eq_ += self.inst["recourse_coeffs"][loc] * r_vars[f"r_{loc + 1}"]
            for clnt in range(self.inst["n_clients"]):
                eq_ += self.inst["client_coeffs"][clnt][loc] * y_vars[f"y_{clnt + 1}_{loc + 1}"]
            c_name = f"capacity_{loc + 1}"
            constrs[c_name] = model.addConstr(eq_ >= 0, name=c_name)

        for clnt in range(self.inst["n_clients"]):
            eq_ = 0
            for loc in range(self.inst["n_locations"]):
                eq_ += y_vars[f"y_{clnt + 1}_{loc + 1}"]
            c_name = f"active_{clnt + 1}"
            constrs[c_name] = model.addConstr(eq_ == scenario[clnt], name=c_name)

        model.update()

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
        for i in range(1, self.inst["n_locations"] + 1):
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

    def get_second_stage_objective(self, sol, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        model = self._make_second_stage_model(scenario)
        model = self.fix_first_stage(model, sol)

        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)
        model.optimize()

        return self.get_second_stage_cost(model)

    def evaluate_first_stage_sol(
        self, sol, scenarios, gap=0.0001, time_limit=3600, threads=1, verbose=0, test_set="0", n_procs=1
    ):
        n_scenarios = len(scenarios)
        scenario_prob = 1 / n_scenarios

        fs_obj = 0
        for i in range(self.inst["n_locations"]):
            fs_obj += self.inst["first_stage_costs"][i] * sol[i]

        pool = Pool(n_procs)

        results = [
            pool.apply_async(
                self.mp_get_second_stage_obj, args=(sol, scenario, scenario_prob, gap, time_limit, threads, verbose)
            )
            for scenario in scenarios
        ]

        results = [r.get() for r in results]

        second_stage_obj_val = np.sum(results)

        return fs_obj + second_stage_obj_val

    def mp_get_second_stage_obj(self, sol, scenario, scenario_prob, gap, time_limit, threads, verbose):
        second_stage_obj = scenario_prob * self.get_second_stage_objective(
            sol, scenario, gap=gap, time_limit=time_limit, threads=threads, verbose=verbose
        )
        return second_stage_obj

    def get_first_stage_extensive_solution(self, model):
        x_sol = {}
        for var in model.getVars():
            if "x" in var.varName and len(var.varName.split("_")) == 2:
                x_sol[var.varName] = np.abs(var.x)
        return x_sol

    def fix_first_stage(self, model, sol):
        for i, var in enumerate(model.getVars()):
            if "x" in var.varName:
                var.ub = sol[i]
                var.lb = sol[i]
        model.update()
        return model

    def get_second_stage_cost(self, model):
        ss_obj = 0
        sol = []
        for var in model.getVars():
            if "y" in var.varName or "r" in var.varName:
                ss_obj += var.obj * var.x
            if "y" in var.varName:
                sol.append(var.x)
        return ss_obj

    def get_scenario_optimal_first_stage(self, scenario, gap=0.0001, time_limit=1e7, threads=1, verbose=0):
        model = self._make_second_stage_model(scenario)
        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("Threads", threads)
        model.optimize()

        sol = {}
        for var in model.getVars():
            v_name = var.varName
            if "x" in var.varName:
                sol[var.varName] = float(int(var.x))

        ss_obj = self.get_second_stage_cost(model)

        return sol, ss_obj

    def get_scenarios(self, n_scenarios, test_set):
        if test_set == "siplib":
            sslp_instance = f'sslp_{self.inst["n_locations"]}_{self.inst["n_clients"]}_{n_scenarios}'
            scenarios = self.inst["siplib_scenario_dict"][sslp_instance]

        else:
            test_set = int(test_set)
            rng = np.random.RandomState()
            rng.seed(n_scenarios + test_set)

            scenarios = rng.randint(0, 2, size=(n_scenarios, self.inst["n_clients"])).tolist()

        return scenarios
