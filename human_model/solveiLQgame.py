import numpy as np
from human_model.iLQgame import *

class ilq_results:
    def __init__(self, xHdims, dynamics, \
                Ps, alphas, costs, \
                u_dim, u_lim, alpha_scale, horizon, max_iteration, tolerence):

        self.ilq_solve = None

        self.xHdims = xHdims
        self.u_dim = u_dim
        self.u_lim = u_lim

        self.dynamics = dynamics

        self.Ps = Ps
        self.alphas = alphas
        self.costs = costs

        self.alpha_scale = alpha_scale
        self.horizon = horizon
        self.max_iteration = max_iteration
        self.tolerence = tolerence

    def solveiLQgame(self, x):
            
        ilq_solver = iLQgame(x, self.u_dim, self.u_lim, self.dynamics, self.costs, self.Ps, self.alphas, \
                            self.alpha_scale, self.horizon, self.max_iteration, self.tolerence)

        ilq_solver.solve()

        self.ilq_solve = ilq_solver


