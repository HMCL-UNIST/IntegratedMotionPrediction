import numpy as np


class ReferenceCost:
    def __init__(self, id, x_ref, vel_ref, u_ref, x_dim, u_dim, Q, R):
        self.id = id
        self.Q = Q
        self.R = R
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.x_ref = x_ref
        self.vel_ref = vel_ref
        self.u_ref = u_ref 

    def calc_r(self, x, u):
        
        x_ref = np.zeros((self.x_dim,1))
        if self.id == 0:
            ref, target_ind = self.x_ref.calc_ref_trajectory_in_T_step(x[:4], 1, self.vel_ref)
            ref_ = ref[-1].reshape(4,1)
            x_ref[:4] = ref_
        elif self.id == 1:
            ref, target_ind = self.x_ref.calc_ref_trajectory_in_T_step(x[4:], 1, self.vel_ref)
            ref_ = ref[-1].reshape(4,1)
            x_ref[4:] = ref_
        else:
            ref, target_ind = self.x_ref.calc_ref_trajectory_in_T_step(x, 1, self.vel_ref)
            ref_ = ref[-1].reshape(4,1)
            x_ref = ref_
            # print( "xbar_D is ", np.round((x - x_ref),2).reshape(len(x),))

        xbar = (x - x_ref)
        
        ubar = (u - self.u_ref)
        r = xbar.reshape(1,self.x_dim)@self.Q@xbar + ubar.reshape(1,self.u_dim)@self.R@ubar
        return r

    def calc_dldx(self, x):

        x_ref = np.zeros((self.x_dim,1))
        if self.id == 0:
            ref, target_ind = self.x_ref.calc_ref_trajectory_in_T_step(x[:4], 1, self.vel_ref)
            ref_ = ref[-1].reshape(4,1)
            x_ref[:4] = ref_
        elif self.id == 1:
            ref, target_ind = self.x_ref.calc_ref_trajectory_in_T_step(x[4:], 1, self.vel_ref)
            ref_ = ref[-1].reshape(4,1)
            x_ref[4:] = ref_
        else:
            ref, target_ind = self.x_ref.calc_ref_trajectory_in_T_step(x, 1, self.vel_ref)
            ref_ = ref[-1].reshape(4,1)
            x_ref = ref_

        drdx = 2*self.Q@(x-x_ref)
        return drdx

    def calc_Hx(self,x):
        d2rd2x = 2*self.Q
        return d2rd2x

    def calc_Hu(self,u):
        d2rdu2 = 2*self.R
        return d2rdu2

class CollisionCost:
    def __init__(self, px_dim, py_dim, x_dim, u_dim, distance):
        
        self.px_dim = px_dim
        self.py_dim = py_dim
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.distance = distance

    def calc_r(self, x):
        

        dist_x = (x[self.px_dim[0]] - x[self.px_dim[1]])
        dist_y = (x[self.py_dim[0]] - x[self.py_dim[1]])
        dist = np.sqrt(dist_x**2 + dist_y**2)

        cost = ( min( (dist-self.distance), 0.0) )**2

        return cost
        
    def calc_dldx(self, x):
        
        dist_x = (x[self.px_dim[0]] - x[self.px_dim[1]])
        dist_y = (x[self.py_dim[0]] - x[self.py_dim[1]])
        dis = np.sqrt(dist_x**2 + dist_y**2) 

        drdx = np.zeros((self.x_dim,1))
        if dis < self.distance:

            drdx[0] = 2*(dist_x)*(dis - self.distance)/dis
            drdx[1] = 2*(dist_y)*(dis - self.distance)/dis
            drdx[4] = -2*(dist_x)*(dis - self.distance)/dis
            drdx[5] = -2*(dist_y)*(dis - self.distance)/dis
        return drdx

    def calc_Hx(self, x):

        dist_x = (x[self.px_dim[0]] - x[self.px_dim[1]])
        dist_y = (x[self.py_dim[0]] - x[self.py_dim[1]])
        dis = np.sqrt(dist_x**2 + dist_y**2) 

        d2rd2x = np.zeros((self.x_dim, self.x_dim))
        if np.sqrt(dist_x**2 + dist_y**2) < self.distance:
            l1 = 2*dist_x**2/dis**2 - 2*dist_x**2*(dis-self.distance)/(dist_x**2 + dist_y**2)**1.5 - self.distance*2/dis + 2
            l2 = -l1
            
            l3 = 2*self.distance*dist_x*dist_y/(dist_x**2 + dist_y**2)**1.5
            l4 = -l3

            l5 = 2*dist_y**2/dis**2 - 2*dist_y**2*(dis-self.distance)/(dist_x**2 + dist_y**2)**1.5 - self.distance*2/dis + 2
            l6 = -l5

            d2rd2x[0,0] = l1 #2*(dist_x)*(dis - 5)/dis
            d2rd2x[0,1] = l3
            d2rd2x[0,4] = l2
            d2rd2x[0,5] = l4

            d2rd2x[1,0] = l3 #2*(dist_y)*(dis - 5)/dis
            d2rd2x[1,1] = l5
            d2rd2x[1,4] = l4
            d2rd2x[1,5] = l6

            d2rd2x[4,0] = l2 #-2*(dist_x)*(dis - 5)/dis
            d2rd2x[4,1] = l4
            d2rd2x[4,4] = l1
            d2rd2x[4,5] = l3

            d2rd2x[5,0] = l4 #-2*(dist_y)*(dis - 5)/dis
            d2rd2x[5,1] = l6
            d2rd2x[5,4] = l3
            d2rd2x[5,5] = l5

        return d2rd2x

    def calc_Hu(self, u):
        return np.zeros((self.u_dim,self.u_dim))

class QuadraticCost:
    def __init__(self, x_dim, u_dim, x_idx, threshold):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.x_idx = x_idx
        self.threshold =threshold
    
    def calc_r(self, x):
        cost = (x[self.x_idx] - self.threshold)**2
        return cost
    
    def calc_dldx(self,x):
        
        drdx = np.zeros((self.x_dim,1))
        drdx[self.x_idx] = 2*(x[self.x_idx] - self.threshold)
        
        return drdx
    
    def calc_Hx(self, x):

        d2rd2x = np.zeros((self.x_dim, self.x_dim))
        d2rd2x[self.x_idx, self.x_idx] = 2

        return d2rd2x
    
    def calc_Hu(self, u):
        return np.zeros((self.u_dim,self.u_dim))

    
class SemiQuadraticCost:
    def __init__(self, x_dim, u_dim, x_idx, threshold, oriented_in):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.x_idx = x_idx
        self.threshold = threshold
        self.oriented_in = oriented_in
    
    def calc_r(self, x):
        
        cost = 0.0
        if self.oriented_in: #state should be smaller than threshold
            if x[self.x_idx] > self.threshold:
                cost = (x[self.x_idx] - self.threshold)**2
            else:
                cost = 0
        else:
            if x[self.x_idx] < self.threshold:
                cost = (self.threshold-x[self.x_idx])**2
            else:
                cost = 0

        return cost
    
    def calc_dldx(self,x):
        
        drdx = np.zeros((self.x_dim,1))
        if self.oriented_in:
            if x[self.x_idx] > self.threshold:
                drdx[self.x_idx] = 2*(x[self.x_idx] - self.threshold)
        else:
            if x[self.x_idx] < self.threshold:
                drdx[self.x_idx] = 2*(self.threshold - x[self.x_idx])
        
        return drdx
    
    def calc_Hx(self, x):

        d2rd2x = np.zeros((self.x_dim, self.x_dim))
        if self.oriented_in:
            if x[self.x_idx] > self.threshold:
                d2rd2x[self.x_idx,self.x_idx] = 2
        else:
            if x[self.x_idx] < self.threshold:
                d2rd2x[self.x_idx,self.x_idx] = -2

        return d2rd2x
    
    def calc_Hu(self, u):
        return np.zeros((self.u_dim,self.u_dim))