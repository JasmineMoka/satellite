from pyomo.environ import *
from pyomo.opt import SolverFactory
import gurobipy as gp
from gurobipy import GRB

# 定义 Pyomo 模型
model = ConcreteModel()
# 假设你已经定义好了模型的变量、参数和约束

# 将 Pyomo 模型转换为 Gurobi 模型
def convert_pyomo_to_gurobi(pyomo_model):
    gurobi_model = gp.Model()

    # 添加变量
    for v in pyomo_model.component_objects(Var, active=True):
        for index in v:
            gurobi_model.addVar(lb=v[index].lb, ub=v[index].ub, name=str(v[index]))

    # 添加约束
    for c in pyomo_model.component_objects(Constraint, active=True):
        for index in c:
            gurobi_model.addConstr(eval(str(c[index].expr)), name=str(c[index]))

    # 添加目标函数
    gurobi_model.setObjective(eval(str(pyomo_model.obj.expr)), GRB.MINIMIZE)

    gurobi_model.update()
    return gurobi_model

# 转换模型
gurobi_model = convert_pyomo_to_gurobi(model)

# 求解模型
gurobi_model.optimize()

# 检查模型是否不可行
if gurobi_model.status == GRB.INFEASIBLE:
    print("模型不可行，正在生成 IIS...")
    gurobi_model.computeIIS()
    gurobi_model.write("model.ilp")

    # 打印 IIS 信息
    for c in gurobi_model.getConstrs():
        if c.IISConstr:
            print(f"不可行约束: {c.constrName}")
