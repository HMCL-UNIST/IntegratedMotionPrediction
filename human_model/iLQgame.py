
import numpy as np
from scipy.linalg import expm, block_diag, pinv

class iLQpoint:
    def __init__(self):
        self.xs = None
        self.us = None
        self.costs = None
        self.Ps = None
        self.alphas = None
        self.Zs = None
        


class iLQgame:
    def __init__(self, x0, u_dim, u_lim, \
                dynamics, costs, Ps, alphas, \
                alpha_scale, horizon, max_iteration, tolerence):

        self.x0 = x0
        self.xdim = len(x0)
        self.u_dim = u_dim
        self.u_lim = u_lim

        self.dynamics = dynamics
        self.Ps = Ps
        self.alphas = alphas
        self.cost = costs
        
        self.num_player = len(costs)
        self.alpha_scale = alpha_scale
        self.horizon = horizon
        self.max_iter = max_iteration 
        self.tolerence = tolerence

        self.current_operation_point = iLQpoint()
        self.last_operating_point = iLQpoint() 
        self.best_operating_point = iLQpoint()  


    def solve(self):
        
        iteration = 0

        Ps = self.Ps
        alphas = self.alphas
        Zs = np.zeros((self.num_player, self.horizon+1, self.xdim, self.xdim ))


        while ( iteration <= self.max_iter and self.is_converged() == False ):
            
            if iteration > 0:
                self.last_operating_point = self.insert_operation_point(self.last_operating_point, xs, us, costs, Ps, Zs, alphas) 

            xs, us, costs = self.compute_operating_point()

            self.current_operation_point = self.insert_operation_point(self.current_operation_point, xs, us, costs, Ps, Zs, alphas)

            As = np.zeros((self.horizon, self.xdim, self.xdim))
            Bs = np.zeros((self.num_player, self.horizon, self.xdim, self.u_dim))

            for k in range(self.horizon):
                A, B = self.dynamics.linearizeDiscrete_Interaction(xs[k], us[k])

                for i in range(self.num_player):
                    As[k] = A
                    Bs[i,k] = B[i]
            
            ls = np.zeros((self.num_player, self.horizon, self.xdim, 1)) #dldx
            Qs = np.zeros((self.num_player, self.horizon, self.xdim, self.xdim)) #Hx
            Rs = np.zeros((self.num_player, self.horizon, self.u_dim, self.u_dim)) #Hu

            for i in range(self.num_player):
                for k in range(self.horizon):
                    us_ik = us[k,self.dynamics.u_dims[i]]
                    total_cost, dldx, Hx, Hu = self.cost[i].quadraticize(xs[k], us_ik)
                    ls[i,k] = dldx
                    Qs[i,k] = Hx
                    Rs[i,k] = Hu

            Ps, alphas, Zs = self.solve_lq_game(As, Bs, ls, Qs, Rs)

            self.Ps = Ps
            self.alphas = alphas



            if self.best_operating_point.xs is None:
                self.best_operating_point = self.insert_operation_point(self.best_operating_point, xs, us, costs, Ps, Zs, alphas)
                best_cost = np.sum(costs)
            elif self.best_operating_point.Zs is None:
                self.best_operating_point = self.insert_operation_point(self.best_operating_point, xs, us, costs, Ps, Zs, alphas)
                best_cost = np.sum(costs)
            elif np.sum(costs) < best_cost and self.current_operation_point.Zs is not None:
                self.best_operating_point = self.insert_operation_point(self.best_operating_point, xs, us, costs, Ps, Zs, alphas)
                best_cost = np.sum(costs)
            
            iteration = iteration + 1
            # print(iteration, " of iteration: ", np.sum(costs))
            # print(self.best_operating_point.us[0])
            
        
        # print("Done: interation is", iteration)

    def insert_operation_point(self, out_, xs, us, costs, Ps, Zs, alphas):
        out_.xs = xs
        out_.us = us
        out_.costs = costs
        out_.Ps = Ps
        out_.Zs = Zs
        out_.alphas = alphas
        return out_

    def solve_lq_game(self, As, Bs, ls, Qs, Rs):
        Ps = np.zeros((self.num_player, self.horizon, self.u_dim, self.xdim))     
        alphas = np.zeros((self.num_player, self.horizon+1, self.u_dim, 1)) 
        Zs =  np.zeros((self.num_player, self.horizon+1,self.xdim, self.xdim))   
        zetas = np.zeros((self.num_player, self.horizon+1, self.xdim, 1))    

        for i in range(self.num_player):
            Zs[i,-1] = Qs[i,-1]
            zetas[i,-1] = ls[i, -1]

        u_dims = self.dynamics.u_dims

        for k in reversed(range(self.horizon)):
            
            for i in range(self.num_player):
                for j in range(self.num_player):
                    if j == 0:
                        s = Rs[i,k] + Bs[i,k].T @ Zs[i, k+1] @ Bs[i,k]
                        Sis = s
                    else:
                        s = Bs[i,k].T @ Zs[i, k+1] @ Bs[j,k]
                        Sis = np.append(Sis,s,1)
                
                if i == 0:
                    S = Sis
                else:
                    S = np.append(S,Sis,0)
            
            for i in range(self.num_player):
                y = Bs[i,k].T @ Zs[i, k+1] @ As[k]
                if i == 0:
                    Y = y
                else:
                    Y = np.append(Y,y,0)
            

            P = pinv(S)@Y
            for i in range(self.num_player):
                Ps[i,k] = P[u_dims[i]]

            
            F = As[k]
            for i in range(self.num_player):
                F = F - Bs[i,k]@Ps[i,k]
            
            for i in range(self.num_player):
                Z = F.T @ Zs[i,k+1] @ F +  Qs[i,k] + Ps[i,k].T@Rs[i,k]@Ps[i,k]
                Zs[i,k] = Z

            for i in range(self.num_player):
                y = Bs[i,k].T @ zetas[i,k+1]
                if i == 0:
                    Y = y
                else:
                    Y = np.append(Y,y,0)

            alpha = pinv(S)@Y
            for i in range(self.num_player):
                alphas[i,k] = alpha[u_dims[i]]
            
            beta = 0
            for i in range(self.num_player):
                beta = beta - Bs[i,k]@alphas[i,k]
            
            for i in range(self.num_player):
                zeta = F.T @ (zetas[i,k+1] + Zs[i,k+1]@beta) + ls[i,k] + Ps[i,k].T@Rs[i,k]@alphas[i,k]
                zetas[i,k] = zeta

        return Ps, alphas, Zs



    def compute_operating_point(self):
        
        xs = np.zeros((self.horizon+1,self.xdim,1))
        xs[0] = self.x0.reshape(self.xdim,1)
        us = np.zeros((self.horizon,self.u_dim*self.num_player,1))
        costs = []

        for k in range(self.horizon):
            if self.current_operation_point.xs is None:
                current_x = np.zeros((self.xdim,1))
                current_u = np.zeros((self.dynamics.u_dim,1))
            else:
                current_x = self.current_operation_point.xs[k]
                current_u = self.current_operation_point.us[k]

            uk = []
            for i in range(self.num_player):
                x_ref = current_x 
                u_id =self.dynamics.u_dims[i]
                u_ref = current_u[u_id]
                P_ik = self.Ps[i,k] # P for i agent at k horizon time
                alpha_ik = self.alphas[i,k]

                if i == 1:
                    scale = 0.01
                else:
                    scale = self.alpha_scale
                ui = get_control(xs[k], x_ref, u_ref, P_ik, alpha_ik, scale)


                #Constriant the control
                ui[0] = max(min(ui[0], self.u_lim[u_id][0]), -self.u_lim[u_id][0])
                ui[1] = max(min(ui[1], self.u_lim[u_id][1]), -self.u_lim[u_id][1])

                uk.append(ui)
            
            uk = np.array(uk)
            uk = uk.reshape(self.u_dim*self.num_player,1)
            us[k] = uk

            cost_k = [] # cost for all agent at k horizon time
            for i in range(self.num_player):
                cost_i = self.cost[i].cost(xs[k], uk[self.dynamics.u_dims[i]])
                cost_k.append(cost_i)
            
            costs.append(cost_k)

            xs[k+1] = self.dynamics.integrate(xs[k], uk)


        return xs, us, costs

    def is_converged(self):

        flag = None

        if self.last_operating_point.xs is None:
            flag = False
        else:
            last_costs = self.last_operating_point.costs
            current_costs = self.current_operation_point.costs

            for i in range(self.num_player):
                last_cost  = np.sum(last_costs[i])
                current_cost = np.sum(current_costs[i]) 

                if np.linalg.norm( (current_cost - last_cost)/last_cost ) > self.tolerence:
                    flag = False
        
        if flag is None:
            flag = True
        
        return flag


def get_control(x, x_ref, u_ref, P, alpha, alpha_scale):

    u = u_ref - P@(x - x_ref) - alpha_scale*alpha

    return u

def get_covariance(RH, BH, Zs):
    Sigma = RH + BH.T@Zs@BH

    return Sigma