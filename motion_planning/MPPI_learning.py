
import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt


import functools
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

def handle_batch_input(func):
    """For func that expect 2D input, handle input that have more than 2 dimensions by flattening them temporarily"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # assume inputs that are tensor-like have compatible shapes and is represented by the first argument
        batch_dims = []
        for arg in args:
            if is_tensor_like(arg) and len(arg.shape) > 2:
                batch_dims = arg.shape[:-1]  # last dimension is type dependent; all previous ones are batches
                break
        # no batches; just return normally
        if not batch_dims:
            return func(*args, **kwargs)

        # reduce all batch dimensions down to the first one
        args = [v.view(-1, v.shape[-1]) if (is_tensor_like(v) and len(v.shape) > 2) else v for v in args]
        ret = func(*args, **kwargs)
        # restore original batch dimensions; keep variable dimension (nx)
        if type(ret) is tuple:
            ret = [v if (not is_tensor_like(v) or len(v.shape) == 0) else (
                v.view(*batch_dims, v.shape[-1]) if len(v.shape) == 2 else v.view(*batch_dims)) for v in ret]
        else:
            if is_tensor_like(ret):
                if len(ret.shape) == 2:
                    ret = ret.view(*batch_dims, ret.shape[-1])
                else:
                    ret = ret.view(*batch_dims)
        return ret

    return wrapper


class mppi_learning:
    def __init__(self, N, K, K2, xlim, ulim, encoder, Q, R, Rd, W, wac = 1, ca_lat_bd=2.5, ca_lon_bd=4):

        #Prams for MPPI
        self.d="cuda"
        self.K = K
        self.K2 = K2
        self.N = N

        self.xlim = xlim
        self.state = None
        self.U = None
        self.states = None
        self.actions = None
        self.u_old = None
        

        self.u_min = torch.tensor([-ulim[2], -ulim[3]]).to(device= self.d)
        self.u_max = torch.tensor([ulim[2], ulim[3]]).to(device= self.d)
        self.u_h_min = torch.tensor([-ulim[0], -ulim[1]]).to(device= self.d)
        self.u_h_max = torch.tensor([ulim[0], ulim[1]]).to(device= self.d)

        u_init = torch.zeros_like(self.u_min)
        self.u_init = u_init.to(self.d)

        self.dtype = torch.float64       

        self.lambda_ = 20

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None

        self.q = Q
        self.r = R
        self.rd = Rd
        self.w = W
        self.wca = wac

        self.ca_lat_bd = ca_lat_bd
        self.ca_lon_bd = ca_lon_bd

        self.encoder = encoder


    @handle_batch_input
    def _dynamics(self, state, u):
        return self.dynamics_update(state, u)
    
    # @handle_batch_input
    # def _dynamics_mc(self, state, u):
    #     return self.dynamics_update_mc(state, u)

    @handle_batch_input
    def _running_cost(self, state, u, prev_u, xh, uh, var):
        return self.running_cost(state, u , prev_u, xh,uh, var)


    def running_cost(self, state, action, prev_action, xh, uh, sig):
       
        cost = torch.zeros(self.K*self.K2).to(device=self.d)
        # state = state.repeat_interleave(self.K2, dim=0)
        action = action.repeat_interleave(self.K2, dim=0)
        

        R_dynamics = self.ilq_results_A.dynamics.subsystems[1]

        cost = self.q[5,5]*((state[:,1]- self.xlim[1]/2)**2) + \
                self.q[6,6]*((state[:,2]- 0.0)**2) + \
                self.q[7,7]*((state[:,3]- self.xlim[3])**2)
        cost -= 4*state[:,0]

        cost += 300*torch.exp(-1.0*((state[:,1]-self.xlim[1])**2)) + \
                300*torch.exp(-1.0*((self.xlim[1]-state[:,1])**2))

      
        idx = state[:, 1] >= self.xlim[1]
        if torch.any(idx):
            cost[idx] += 10000

        idx = state[:, 1] <= -self.xlim[1]
        if torch.any(idx):
            cost[idx] += 10000
        
        idx = state[:, 3] <= 0.0
        if torch.any(idx):
            cost[idx] += 10000

        idx = state[:, 3] >= self.xlim[7]
        if torch.any(idx):
            cost[idx] += 10000
        
        psi_h = xh[:,2]
        dist_x =  (xh[:,0]-state[:,0])*torch.cos(psi_h) + (xh[:,1]-state[:,1])*torch.sin(psi_h)
        dist_y =  (xh[:,0]-state[:,0])*torch.sin(psi_h) - (xh[:,1]-state[:,1])*torch.cos(psi_h)

        if self.beta_w:
            var = torch.zeros(self.K*self.K2,2).to(device=self.d)
            var[:,0] = (torch.cos(xh[:,2])**2 * sig[:,0] + torch.sin(xh[:,2])**2  * sig[:,1])
            var[:,1] = (torch.sin(xh[:,2])**2 * sig[:,0] + torch.cos(xh[:,2])**2  * sig[:,1])

            mu_a = (var[:,0] + self.ca_lon_bd).to(self.d)
            mu_b = (var[:,1] + self.ca_lat_bd).to(self.d)
        else:
            mu_a = self.ca_lon_bd*torch.ones(self.K*self.K2).to(self.d)
            mu_b = self.ca_lat_bd*torch.ones(self.K*self.K2).to(self.d)

        self.initial = True

        cost += self.w*torch.exp(-1/2*self.wca*(dist_x**2/(mu_a**2) + dist_y**2/(mu_b**2))) 
        idx = (dist_x**2/(mu_a**2) + dist_y**2/(mu_b**2)) <= 1
        if torch.any(idx):
            cost[idx] += 10000

        cost += self.r[0,0]*(action[:,0]**2) + self.r[1,1]*(action[:,1]**2)

        if prev_action is not None:
            prev_action = prev_action.repeat_interleave(self.K2, dim=0)
            cost += self.rd[0,0]*((action[:,0] - prev_action[:,0])**2) + self.rd[1,1]*((action[:,1] - prev_action[:,1])**2)

        return cost

    def _compute_human_action(self, xR):

        xH_A, xH_D, u_A, u_D, Sigma_A, Sigma_D = self.update_human_state(xR)
        variances = torch.zeros(self.K*self.K2, self.N, 4, device=self.d, dtype=self.dtype)

        if self.active:
            coin = np.random.uniform(0,1,self.K2)
            coin_A = len( np.where(coin <= self.theta_init)[0] )
            
            if self.beta_w:
                n_sampled = 20
            else:
                n_sampled = coin_A

            Sigma_A[:,:,0,1] = 0.0
            Sigma_A[:,:,1,0] = 0.0
            Sigma_D[:,:,0,1] = 0.0
            Sigma_D[:,:,1,0] = 0.0

            ## Sample for Attentive Human Driver 
            uH = torch.zeros((self.K*self.K2, self.N, self.Nu), device=self.d, dtype=self.dtype)
            xH = torch.zeros((self.K*self.K2 , self.N, 4), device=self.d, dtype=self.dtype)
            uH_sampled = torch.zeros((self.K*n_sampled, self.N, self.Nu), device=self.d, dtype=self.dtype)

            uH_A_dist_ac = Normal(u_A[:,:,0], torch.sqrt(Sigma_A[:,:,0,0]))
            uH_A_dist_de1 = Normal(u_A[:,:,1], torch.sqrt(Sigma_A[:,:,1,1]))
            
            #[[u1_sampled, u2_sampled, ... u{coinA}_sampled], [u1_sampled, u2_sampled, ... u{coinA}_sampled], ... # of K]: [self.K2*self.K, 2]
            perturbed_uH_A_ac = uH_A_dist_ac.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N) 
            perturbed_uH_A_del = uH_A_dist_de1.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N)
            perturbed_uH_A_ac = torch.max(torch.min(perturbed_uH_A_ac, self.u_h_max[0]), self.u_h_min[0])
            perturbed_uH_A_del = torch.max(torch.min(perturbed_uH_A_del, self.u_h_max[1]), self.u_h_min[1])
            
            uH[:self.K*coin_A,:,0] = perturbed_uH_A_ac[:self.K*coin_A] 
            uH[:self.K*coin_A,:,1] = perturbed_uH_A_del[:self.K*coin_A] 
            
            if coin_A != 0:
                uH_sampled[:,:,0] = perturbed_uH_A_ac
                uH_sampled[:,:,1] = perturbed_uH_A_del

                xH_sampled = self.trajectory_update(self.x0[:4], uH_sampled)
                xH[:self.K*coin_A] = xH_sampled[:self.K*coin_A]
                var_A = torch.std(xH_sampled, dim=0)
                variances[:self.K*coin_A] = var_A.repeat(self.K*coin_A,1).reshape(self.K*coin_A,self.N,-1)

            ## Sample for Attentive Human Driver 
            
            if self.beta_w:
                n_sampled = 20
            else:
                n_sampled = self.K2-coin_A

            uH_sampled = torch.zeros((self.K*n_sampled, self.N, self.Nu), device=self.d, dtype=self.dtype)
            coin_D = self.K2-coin_A

            uH_D_dist_ac = Normal(u_D[:,:,0], torch.sqrt(Sigma_D[:,:,0,0]))
            uH_D_dist_de1 = Normal(u_D[:,:,1], torch.sqrt(Sigma_D[:,:,1,1]))
            
            perturbed_uH_D_ac = uH_D_dist_ac.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N) 
            perturbed_uH_D_del = uH_D_dist_de1.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N)
            perturbed_uH_D_ac = torch.max(torch.min(perturbed_uH_D_ac, self.u_h_max[0]), self.u_h_min[0])
            perturbed_uH_D_del = torch.max(torch.min(perturbed_uH_D_del, self.u_h_max[1]), self.u_h_min[1])
            
            uH[self.K*coin_A:,:,0] = perturbed_uH_D_ac[:self.K*coin_D]
            uH[self.K*coin_A:,:,1] = perturbed_uH_D_del[:self.K*coin_D]
            
            if coin_D != 0:
                uH_sampled[:,:,0] = perturbed_uH_D_ac
                uH_sampled[:,:,1] = perturbed_uH_D_del

                xH_sampled = self.trajectory_update(self.x0[:4], uH_sampled)
                xH[self.K*coin_A:] = xH_sampled[:self.K*coin_D]
                var_D = torch.std(xH_sampled, dim=0)
                variances[self.K*coin_A:] = var_D.repeat(self.K*coin_D,1).reshape(self.K*coin_D,self.N,-1)

        else:
            if self.theta_init >= 0.5:
                if self.beta_w:
                    n_sampled = 20
                else:
                    n_sampled = self.K2

                Sigma_A[:,:,0,1] = 0.0
                Sigma_A[:,:,1,0] = 0.0

                uH = torch.zeros((self.K*self.K2, self.N, self.Nu), device=self.d, dtype=self.dtype)
                xH = torch.zeros((self.K*self.K2 , self.N, 4), device=self.d, dtype=self.dtype)
                uH_sampled = torch.zeros((self.K*n_sampled, self.N, self.Nu), device=self.d, dtype=self.dtype)

                uH_A_dist_ac = Normal(u_A[:,:,0], torch.sqrt(Sigma_A[:,:,0,0]))
                uH_A_dist_de1 = Normal(u_A[:,:,1], torch.sqrt(Sigma_A[:,:,1,1]))
                
                #[[u1_sampled, u2_sampled, ... u{coinA}_sampled], [u1_sampled, u2_sampled, ... u{coinA}_sampled], ... # of K]: [self.K2*self.K, 2]
                perturbed_uH_A_ac = uH_A_dist_ac.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N) 
                perturbed_uH_A_del = uH_A_dist_de1.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N)
                perturbed_uH_A_ac = torch.max(torch.min(perturbed_uH_A_ac, self.u_h_max[0]), self.u_h_min[0])
                perturbed_uH_A_del = torch.max(torch.min(perturbed_uH_A_del, self.u_h_max[1]), self.u_h_min[1])
                
                uH[:self.K*self.K2,:,0] = perturbed_uH_A_ac[:self.K*self.K2]
                uH[:self.K*self.K2,:,1] = perturbed_uH_A_del[:self.K*self.K2]
                uH_sampled[:,:,0] = perturbed_uH_A_ac
                uH_sampled[:,:,1] = perturbed_uH_A_del

                xH_sampled = self.trajectory_update(self.x0[:4], uH_sampled)
                xH[:self.K*self.K2] = xH_sampled[:self.K*self.K2]
                var_A = torch.std(xH_sampled, dim=0)
                variances = var_A.repeat(self.K*self.K2,1).reshape(self.K*self.K2,self.N,-1)
                
                coin_A =  self.K2
            else:

                if self.beta_w:
                    n_sampled = 20
                else:
                    n_sampled = self.K2

                Sigma_D[:,:,0,1] = 0.0
                Sigma_D[:,:,1,0] = 0.0

                uH = torch.zeros((self.K*self.K2, self.N, self.Nu), device=self.d, dtype=self.dtype)
                xH = torch.zeros((self.K*self.K2 , self.N, 4), device=self.d, dtype=self.dtype)
                uH_sampled = torch.zeros((self.K*n_sampled, self.N, self.Nu), device=self.d, dtype=self.dtype)

                uH_D_dist_ac = Normal(u_D[:,:,0], torch.sqrt(Sigma_D[:,:,0,0]))
                uH_D_dist_de1 = Normal(u_D[:,:,1], torch.sqrt(Sigma_D[:,:,1,1]))

                perturbed_uH_D_ac = uH_D_dist_ac.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N) 
                perturbed_uH_D_del = uH_D_dist_de1.sample((n_sampled,)).transpose(0,1).reshape(-1,self.N)
                perturbed_uH_D_ac = torch.max(torch.min(perturbed_uH_D_ac, self.u_h_max[0]), self.u_h_min[0])
                perturbed_uH_D_del = torch.max(torch.min(perturbed_uH_D_del, self.u_h_max[1]), self.u_h_min[1])
                
                uH[:self.K*self.K2,:,0] = perturbed_uH_D_ac[:self.K*self.K2]
                uH[:self.K*self.K2,:,1] = perturbed_uH_D_del[:self.K*self.K2]
                uH_sampled[:,:,0] = perturbed_uH_D_ac
                uH_sampled[:,:,1] = perturbed_uH_D_del


                xH_sampled = self.trajectory_update(self.x0[:4], uH_sampled)
                xH[:self.K*self.K2] = xH_sampled[:self.K*self.K2]
                var_D = torch.std(xH_sampled, dim=0)
                variances = var_D.repeat(self.K*self.K2,1).reshape(self.K*self.K2,self.N,-1)

                coin_A = 0

        return xH, xH_A, xH_D, uH, u_A, u_D, Sigma_A, Sigma_D, variances, coin_A

    def updateTheta(self, theta_now, uH, uH_A, uH_D, u_A_sigma, u_D_sigma):
        
        if self.beta_w:
            x_next_sig_a = u_A_sigma/(self.beta.a.trunc_mu)
            x_next_sig_d = u_D_sigma/(self.beta.d.trunc_mu)
        else:
            x_next_sig_a = u_A_sigma
            x_next_sig_d = u_D_sigma

        pdf_a = self.calc_pdf(uH, uH_A, x_next_sig_a)
        pdf_d = self.calc_pdf(uH, uH_D, x_next_sig_d)
        denom = theta_now*pdf_a
        denom2 = (torch.ones_like(theta_now)-theta_now)*pdf_d

        theta_ = denom/(denom+denom2)
        # theta_ = theta_*0.5 + theta_now*0.5

        return theta_ 

    def calc_pdf(self, u, u_pred, u_sigma):
        
        u_pred = u_pred.repeat_interleave(self.K2, dim=0)
        u_sigma = u_sigma.repeat_interleave(self.K2, dim=0)

        
        a = (u_pred[:,0]-u[:,0])**2
        pdf_u_acc = torch.exp(-0.5*( (u_pred[:,0]-u[:,0])**2/u_sigma[:,0,0] ))/torch.sqrt(u_sigma[:,0,0]*2*math.pi)
        pdf_u_del = torch.exp(-0.5*( ((u_pred[:,1]-u[:,1])**2).T@(1/u_sigma[:,1,1]) )) /torch.sqrt(u_sigma[:,1,1]*2*math.pi)

        pdf = (pdf_u_acc + pdf_u_del)/2
        pdf = pdf.to(device =  self.d, dtype= self.dtype)

        return pdf
     

    def solve_mppi_learning(self, state, RH, ilq_results_A, ilq_results_D, theta_prob, beta_distr, t=None, Human_seq=None, learning=False):
        
        
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state[4:].to(dtype=self.dtype, device=self.d)
        self.x0 = state

        self.beta = beta_distr

        self.theta_init = theta_prob[0]
        self.theta_prob = torch.zeros((self.N,self.K*self.K2), device=self.d, dtype=self.dtype)
        self.theta_prob_sampled = torch.zeros((self.N,self.K*self.K2), device=self.d, dtype=self.dtype)
        self.theta_prob[0,:] = theta_prob[0]

        self.ilq_results_A = ilq_results_A
        self.ilq_results_D = ilq_results_D

        self.Human_history = Human_seq.unsqueeze(0).repeat_interleave((self.K*self.K2), dim=0)
        self.current_t = t

        if self.U is None:
            self.U = torch.torch.distributions.Uniform(low=self.u_min, high=self.u_max).sample((self.N,))
        else:
            self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
    
        self.RH = RH    

        self.active = active
        self.beta_w = beta_w

        states, xH, num_sampled_A, cost_total, cost_total2 = self._compute_total_cost_batch()

        if torch.any(torch.isnan(cost_total2)):

            beta = torch.min(cost_total)
            self.cost_total_non_zero = torch.where(torch.isnan(cost_total),  torch.zeros(1).to(device=self.d, dtype=self.dtype), _ensure_non_zero(cost_total, beta, 1 / self.lambda_)) 
        else:
            beta = torch.min(cost_total)
            self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)


        uR = self.perturbed_action.repeat_interleave(self.K2, dim=0)
        if torch.all(torch.isnan(cost_total2)):
            beta_id = torch.argmin(cost_total)
            self.U = uR[beta_id] 
            # self.U = torch.zeros(self.N,2)
            print("NO SOLUSTION")
        else:
            if self.active:
                theta_prob_prod = torch.prod(self.theta_prob_sampled, 0)
                weight = theta_prob_prod*self.cost_total_non_zero
                eta = torch.sum(weight)
                weight = weight/eta
                self.omega = torch.where(torch.isnan(weight),  torch.zeros(1).to(dtype=self.dtype, device=self.d), weight)
                for t in range(self.N):                  
                    # self.theta_prob_sampled[t] = self.theta_prob_sampled[t]/torch.sum(self.theta_prob_sampled[t])
                    self.U[t] = torch.sum( self.omega.view(-1, 1)*uR[:,t], dim=0)    
            else:
                eta = torch.sum(self.cost_total_non_zero)
                self.omega = (1. / eta) * self.cost_total_non_zero
                self.omega = torch.where(torch.isnan(self.omega),  torch.zeros(1).to(dtype=self.dtype, device=self.d), self.omega)  
                for t in range(self.N):  
                    self.U[t] = torch.sum(self.omega.view(-1, 1)*uR[:,t], dim=0)

        # print(self.omega)
        xR = self.trajectory_update(state[4:], self.U.unsqueeze(0))
        return xR.squeeze(0).cpu().numpy(), states.cpu().numpy(), xH.cpu().numpy(), num_sampled_A, self.U[0].cpu().numpy(), self.U.cpu().numpy()


    def _compute_rollout_costs(self, perturbed_actions):
        K, T, Nu = perturbed_actions.shape

        cost_total = torch.zeros((K,), device=self.d, dtype=self.dtype)
        cost_samples = torch.zeros((K,), device=self.d, dtype=self.dtype)
        cost_const =  torch.zeros((K,), device=self.d, dtype=self.dtype)
        cost_var = torch.zeros_like(cost_total)

        self.Nx = self.ilq_results_A.dynamics.x_dim
        self.Nu = Nu
        self.dt = self.ilq_results_A.dynamics.dt
        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.Nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)


        states = []
        actions = []

        for t in range(self.N):
            u =  perturbed_actions[:, t]
            state = self._dynamics(state, u)
            actions.append(u)
            states.append(state)
        actions = torch.stack(actions)
        states = torch.stack(states)
        
        xH, xH_A, xH_D, uH, u_A, u_D, self.Sigma_A, self.Sigma_D, var, num_sampled_A = self._compute_human_action(states)
        
        prev_u =None

        c = torch.zeros((self.K*self.K2), device=self.d, dtype=self.dtype)
        c2 = torch.zeros((self.K*self.K2), device=self.d, dtype=self.dtype)
        states = states.repeat_interleave(self.K2, dim=1)
        # Human_seq = torch.zeros((self.K*self.K2, self.N, 10)).to(torch.device("cuda"))

        for t in range(self.N):
            seq = torch.cat((torch.cat((xH[:,t], states[t]), dim=1), uH[:,t]), dim=1)
            seq = seq.unsqueeze(1)
            if self.current_t > 0:
                if t == 0:
                    Human_seq = torch.cat((self.Human_history[:,-self.current_t:],seq), dim=1)
                else:
                    Human_seq = torch.cat((Human_seq,seq), dim=1)
            else:
                if t == 0:
                    Human_seq = seq
                else:
                    Human_seq = torch.cat((Human_seq,seq), dim=1)

            _, seq_len, _= Human_seq.shape        
              
            if seq_len < self.N:
                self.theta_prob[t] = self.theta_init
                self.theta_prob_sampled[t,:self.K*num_sampled_A] = self.theta_init
                self.theta_prob_sampled[t, self.K*num_sampled_A:] = 1- self.theta_init
            else:
                if t == 0: 
                    self.theta_prob_sampled[t,:self.K*num_sampled_A] = self.theta_init
                    self.theta_prob_sampled[t, self.K*num_sampled_A:] = 1- self.theta_init
                else:
                    intention, intention_prob = self.encoder.get_internalstate(Human_seq[:,-self.N:].to(torch.float32))
                    self.theta_prob[t] = intention_prob.squeeze(1)
                    self.theta_prob_sampled[t, :self.K*num_sampled_A] = self.theta_prob[t, :self.K*num_sampled_A]
                    self.theta_prob_sampled[t, self.K*num_sampled_A:] = torch.ones(self.K*(self.K2-num_sampled_A)).to(self.d) - self.theta_prob[t, self.K*num_sampled_A:]
                
            c += self._running_cost(states[t], actions[t], prev_u, xH[:,t], uH[:,t], var[:,t] ) 

            prev_u = actions[t]
        
        if any(c >= 10000):
            c2 = torch.where( c >= 10000, torch.tensor(float('nan')).to(device=self.d, dtype=self.dtype), c)
        else:
            c2 = c

        return states, xH, num_sampled_A, c, c2

    def _compute_total_cost_batch(self):
        
        ##Ego vehicle action sampling
        noise = torch.torch.distributions.Uniform(low=self.u_min, high=self.u_max).sample((self.K, self.N))
        self.perturbed_action = noise
        states, xH, num_sampled_A, cost_total, cost_total2 = self._compute_rollout_costs(self.perturbed_action)


        return states, xH, num_sampled_A, cost_total, cost_total2

    def _slice_control(self, t):
        return slice(t * self.Nu, (t + 1) * self.Nu)

    def dynamics_update(self,x,u):
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(device=self.d)    
        if not torch.is_tensor(u):
            u = torch.tensor(u).to(device=self.d)    
        

        if x.dim() > 1:
            delta = u[:,1]
            a = u[:,0]
            L = self.ilq_results_A.dynamics.subsystems[0].L

            nx = torch.clone(x).to(device=self.d)  
            v = x[:,3]
            psi = x[:,2]

            nx[:,0] = nx[:,0] + v * torch.cos(psi) * self.dt
            nx[:,1] = nx[:,1] + v * torch.sin(psi) * self.dt
            nx[:,2] = nx[:,2] + v * torch.tan(delta) * self.dt / L
            nx[:,3] = nx[:,3] + a * self.dt
        else:
            delta = u[1]
            a = u[0]
            L = self.ilq_results_A.dynamics.subsystems[0].L

            nx = torch.clone(x).to(device=self.d)  
            v = x[3]
            psi = x[2]

            nx[0] = x[0] + v * torch.cos(x[2]) * self.dt
            nx[1] = x[1] + x[3]* torch.sin(x[2]) * self.dt
            nx[2] = x[2] + x[3] * torch.tan(x[2]) * self.dt / L
            nx[3] = x[3] + a * self.dt

        return nx
    
    def dynamics_update_mc(self,x,u):
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(device=self.d)    
        if not torch.is_tensor(u):
            u = torch.tensor(u).to(device=self.d)    
        


        delta = u[:,1]
        a = u[:,0]
        L = self.ilq_results_A.dynamics.subsystems[0].L

        nx = torch.clone(x).to(device=self.d)  
        # v = x[:,3]
        # psi = x[:,2]

        
        
        nx[:,0] = nx[:,0] + nx[:,3]  * torch.cos(nx[:,2]) * self.dt
        nx[:,1] = nx[:,1] + nx[:,3]  * torch.sin(nx[:,2]) * self.dt
        nx[:,2] = nx[:,2] + nx[:,3] * torch.tan(delta) * self.dt / L
        nx[:,3] = nx[:,3] + a * self.dt

        return nx

    def trajectory_update(self,x,u):
        if not torch.is_tensor(x):
            x = torch.tensor(x).to(device=self.d)    
        if not torch.is_tensor(u):
            u = torch.tensor(u).to(device=self.d)    
        
        L = self.ilq_results_A.dynamics.subsystems[0].L
        state = torch.zeros((u.shape[0], self.N+1, 4)).to(self.d)
        state[:,0] = x

        for n in range(self.N):
            state[:,n+1,0] = state[:,n,0] + state[:,n,3]* torch.cos(state[:,n,2]) * self.dt
            state[:,n+1,1] = state[:,n,1] + state[:,n,3] * torch.sin(state[:,n,2]) * self.dt
            state[:,n+1,2] = state[:,n,2] + state[:,n,3] * torch.tan(u[:,n,1]) * self.dt /L
            state[:,n+1,3] = state[:,n,3] +  u[:,n,0]* self.dt
        
        # state[:,1:,0] = state[:,:-1,0] + state[:,:-1,3]* torch.cos(state[:,:-1,2]) * self.dt
        # state[:,1:,1] = state[:,:-1,1] + state[:,:-1,3] * torch.sin(state[:,:-1,2]) * self.dt
        # state[:,1:,2] = state[:,:-1,2] + state[:,:-1,3] * torch.tan(u[:,:,1]) * self.dt /L
        # state[:,1:,3] = state[:,:-1,3] +  u[:,:,0]* self.dt


        return state[:,1:]

    
    def linearizeDiscrete(self, x, u, L, dt, player_num):

        Bc = torch.zeros((self.K, self.N, 4*player_num, 2), dtype=self.dtype, device=self.d)
        Bc[:,:,2,1] = x[:,:,3]*dt/((torch.cos(u[:,:,1])**2) * L)
        Bc[:,:,3,0] = 1.0*dt
            
        return Bc


    def update_human_state(self, xR):
        
        ###GET ILQ results
        x_ilq_ref_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.xs).to(self.d)
        u_ilq_ref_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.us).to(self.d)

        x_ilq_ref_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.xs).to(self.d)
        u_ilq_ref_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.us).to(self.d)

        Ps_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.Ps).to(self.d)
        alphas_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.alphas).to(self.d)
        Z_A = torch.tensor(self.ilq_results_A.ilq_solve.best_operating_point.Zs).to(self.d)

        Ps_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.Ps).to(self.d)
        alphas_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.alphas).to(self.d)
        Z_D = torch.tensor(self.ilq_results_D.ilq_solve.best_operating_point.Zs).to(self.d)

        alpha_scale = self.ilq_results_A.ilq_solve.alpha_scale
        
        ### Attentive
        states = torch.zeros((self.K, self.N + 1,  self.Nx), dtype=self.dtype, device=self.d)
        uH = torch.zeros((self.K, self.N,  self.Nu), dtype=self.dtype, device=self.d)
        states[:,0] = self.x0
        for i in range(self.N):
            uH_ = u_ilq_ref_A[i,:2] - Ps_A[0,i]@(states[:,i]- x_ilq_ref_A[i].T).T - 0.01*alphas_A[0,i]
            uH[:,i] = uH_.T
            states[:, i+1, :4] = self._dynamics(states[:,i,:4], uH_.T )
            states[:, i+1, 4:] = xR[i]

        xH_A = states[:,:,:4]
        u_A =  torch.zeros((self.K, self.N,  4), dtype=self.dtype, device=self.d)
        u_A[:,:,:2] = uH
        u_A[:,:,2:] = self.U
        B = self.linearizeDiscrete(states[:,:-1], u_A, self.ilq_results_A.dynamics.subsystems[0].L, self.dt, 2 )
        Sigma_A = self.get_covariance(self.RH, B, Z_A[:,:self.N] )
        
        ### Distracted
        states = torch.zeros((self.K, self.N + 1, 4), dtype=self.dtype, device=self.d)
        uH = torch.zeros((self.K, self.N,  self.Nu), dtype=self.dtype, device=self.d)
        states[:,0] = self.x0[:4]
        for i in range(self.N):
            uH_ = u_ilq_ref_D[i,:2] - Ps_D[0,i]@(states[:,i]- x_ilq_ref_D[i].T).T - 0.01*alphas_D[0,i]
            uH[:,i] = uH_.T
            states[:,i + 1, :4] = self._dynamics(states[:,i,:4], uH_.T )

        xH_D = states[:,:,:4]
        u_D =  torch.zeros((self.K, self.N,  2), dtype=self.dtype, device=self.d)
        u_D[:,:,:2] = uH
        B = self.linearizeDiscrete(states[:,:-1], u_D, self.ilq_results_A.dynamics.subsystems[0].L, self.dt, 2 )
        Sigma_D = self.get_covariance(self.RH, B[:,:,:4], Z_D[:,:self.N] )

        # states = torch.zeros((self.K, self.N + 1,  self.Nx), dtype=self.dtype, device=self.d)
        # uH = torch.zeros((self.K, self.N,  self.Nu), dtype=self.dtype, device=self.d)
        # states[:,0] = self.x0
        # for i in range(self.N):
        #     uH_ = u_ilq_ref_D[i,:2] - Ps_D[0,i]@(states[:,i]- x_ilq_ref_D[i].T).T - 0.01*alphas_D[0,i]
        #     uH[:,i] = uH_.T
        #     states[:, i+1, :4] = self._dynamics(states[:,i,:4], uH_.T )
        #     states[:, i+1, 4:] = xR[i]

        # xH_D = states[:,:,:4]
        # u_D =  torch.zeros((self.K, self.N,  2), dtype=self.dtype, device=self.d)
        # u_D[:,:,:2] = uH
        # B = self.linearizeDiscrete(states[:,:-1], u_D, self.ilq_results_A.dynamics.subsystems[0].L, self.dt, 2 )
        # Sigma_D = self.get_covariance(self.RH, B, Z_D[:,:self.N] )

        return xH_A, xH_D, u_A, u_D, torch.abs(Sigma_A), torch.abs(Sigma_D)

    def get_covariance(self, RH, B, Zs ):
        Z = Zs[0]
        
        RH = torch.tensor(RH).to(self.d)
        Sigma = torch.zeros((self.K, self.N,  self.Nu, self.Nu), dtype=self.dtype, device=self.d)
        
        Bi = B.transpose(2, 3)
        Z = Z.unsqueeze(0)
        Zs = Z.repeat(self.K,1,1,1)
        Sigma = torch.inverse(RH + Bi @ Zs @ B)

        return Sigma
    
