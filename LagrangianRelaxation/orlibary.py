import numpy as np
class ORLibary:
    def __init__(self):
        self.route = './data\\'

    def gen_assign_prob(self, file_name):
        with open(self.route + file_name, 'r') as f:
            gen_assign_prob = f.read()

        gen_assign_prob = gen_assign_prob.strip().split('\n')
        size = gen_assign_prob[0].strip().split(' ')

        n_bs_number, n_per_cell_number, n_leo_number = int(size[0]), int(size[1]), int(size[2])
        user_number = n_bs_number * n_per_cell_number

        bs_compute_capacity = gen_assign_prob[-2].strip().split(' ')
        leo_compute_capacity = gen_assign_prob[-1].strip().split(' ')


        # 存储基站和卫星计算能力约束
        # bs_compute_capacity = [int(n_bs_number) for i in range(n_bs_number)]
        # leo_compute_capacity = [int(n_leo_number) for i in range(n_leo_number)]

        # T ij max-》A
        deadline = []
        for i in range(1, len(gen_assign_prob )):
            temp = gen_assign_prob [i].strip().split(' ')
            temp = [int(x) for x in temp]
            for x in temp:
                deadline.append(x)
            if len(deadline) == user_number:
                break
            if len(deadline) > user_number:
                print("constraints data error")

        deadline = np.array(deadline).reshape((n_bs_number, n_per_cell_number))
        # cost_coeff = A[0:n_bs_number * n_per_cell_number, :]
        # deadline = deadline[n_bs_number * n_per_cell_number:, :]
        # return n_bs_number * n_per_cell_number, n_leo_number, deadline, b, cost_coeff, bs_compute_capacity, leo_compute_capacity
        return n_bs_number, n_per_cell_number, n_leo_number, deadline, bs_compute_capacity, leo_compute_capacity