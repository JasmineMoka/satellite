from pyomo.environ import *

# 创建具体模型
model = ConcreteModel()

# 定义索引集
model.M = RangeSet(1, 4)
model.N = RangeSet(1, 4)
model.O = RangeSet(1, 3)

# 定义变量
model.alpha = Var(model.M, model.N, within=Binary)
model.beta = Var(model.M, model.N, within=Binary)
model.gamma = Var(model.M, model.N, within=Binary)
model.rho = Var(model.M, model.N, model.O, within=Binary)
model.eta = Var(model.M, model.N, model.O, within=Binary)
model.epsilon = Var(model.M, model.N, within=NonNegativeReals, bounds=(0, 1))
model.delta = Var(model.M, model.N, model.O, within=NonNegativeReals, bounds=(0, 1))
model.f_bs = Var(model.M, model.N, within=NonNegativeReals)
model.f_leo = Var(model.M, model.N, model.O, within=NonNegativeReals)

# 定义参数并初始化
def initialize_deadline(model, i, j):
    return 10  # 示例值

def initialize_bs_compute_capacity(model, i):
    return 100  # 示例值

def initialize_leo_compute_capacity(model, k):
    return 200  # 示例值

model.deadline = Param(model.M, model.N, initialize=initialize_deadline)
model.bs_compute_capacity = Param(model.M, initialize=initialize_bs_compute_capacity)
model.leo_compute_capacity = Param(model.O, initialize=initialize_leo_compute_capacity)
model.Cache = Param(model.O, initialize=3 * pow(10, 4))
model.compute_capacity_user = Param(model.M, model.N, initialize=pow(10, 5))
model.f_LEO = Param(model.O, initialize=lambda model, k: 100)

# 定义目标函数
def objective_rule(model):
    return sum(model.f_bs[i, j] * model.deadline[i, j] for i in model.M for j in model.N)

model.objective = Objective(rule=objective_rule, sense=minimize)

# 定义约束
def constraint_rule_1(model, i, j):
    return model.f_bs[i, j] <= model.bs_compute_capacity[i]

model.constraint1 = Constraint(model.M, model.N, rule=constraint_rule_1)

def constraint_rule_2(model, i, j, k):
    return model.f_leo[i, j, k] <= model.leo_compute_capacity[k]

model.constraint2 = Constraint(model.M, model.N, model.O, rule=constraint_rule_2)

# 运行求解器
solver = SolverFactory('gurobi')
result = solver.solve(model, tee=True)

# 打印结果
model.display()
