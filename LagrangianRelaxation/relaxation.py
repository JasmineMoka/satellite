from pyomo.core import Objective
from scipy.optimize import minimize


# class Relaxation:
#
#     def __init__(self, model, relaxed_constrs):
#         self.model = model
#         self.relaxed_constrs = relaxed_constrs
#         self.lambdas = {constr: 0.0 for constr in relaxed_constrs}
#         self.subgrad = None
#         self.step_size = 1.0
#
#
#     def relax_constrs(self):
#         for constr in self.relaxed_constrs:
#             constr.deactivate()
#
#     def set_objective(self, relaxed_obj_rule):
#             # 定义拉格朗日目标函数
#             def lagrangian_obj_rule(model):
#                 lagrangian_term = sum(self.lambdas[constr] * (constr() - (constr.lower if constr.lower is not None else 0)) for constr in self.relaxed_constrs)
#                 return relaxed_obj_rule(model) + lagrangian_term
#
#             self.model.obj = Objective(rule=lagrangian_obj_rule, sense=minimize)
#
#
#     def compute_subgrad(self):
#         # 计算子梯度
#         self.subgrad = {constr: constr() - 1 for constr in self.relaxed_constrs}
#
#     def update_lambdas(self):
#         # 更新拉格朗日乘子
#         for constr in self.relaxed_constrs:
#             self.lambdas[constr] += self.step_size * self.subgrad[constr]
#
#     def get_step_size(self, iteration, max_iter):
#         # 逐渐减小的步长
#         self.step_size = 1.0 / (iteration + 1)
#         return self.step_size

class Relaxation:
    def __init__(self, model, relaxed_constrs):
        self.model = model
        self.relaxed_constrs = relaxed_constrs
        self.lambdas = {constr: 1.0 for constr in relaxed_constrs}
        self.subgrad = None
        self.step_size = 1.0

    def relax_constrs(self):
        for constr in self.relaxed_constrs:
            constr.deactivate()

    def set_objective(self, relaxed_obj_rule):
        # 定义拉格朗日目标函数
        def lagrangian_obj_rule(model):
            lagrangian_term = sum(
                self.lambdas[constr] * (constr() - (constr.lower if constr.lower is not None else 0))
                for constr in self.relaxed_constrs
            )
            return relaxed_obj_rule(model) + lagrangian_term

        self.model.obj = Objective(rule=lagrangian_obj_rule, sense=minimize)
    def compute_subgrad(self):
        # 计算子梯度
        self.subgrad = {constr: constr() - (constr.lower if constr.lower is not None else 0) for constr in self.relaxed_constrs}

    def update_lambdas(self):
        # 更新拉格朗日乘子
        for constr in self.relaxed_constrs:
            self.lambdas[constr] += self.step_size * self.subgrad[constr]

    def get_step_size(self, iteration, max_iter):
        # 逐渐减小的步长
        self.step_size = 1 / (iteration + 1)
        return self.step_size



