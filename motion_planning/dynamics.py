import numpy as np
import math
from scipy.linalg import expm, block_diag, pinv


class InteractionDynamics:
    def __init__(self, dt, x_dim, u_dim, x_dims, u_dims, subsystems):

        self.dt = dt
        self.x_dim = x_dim # number of state for all agents
        self.x_dims = x_dims # np.array
        self.u_dim = u_dim # number of input for all agents
        self.u_dims = u_dims # np.array
        self.num_player = len(subsystems)
        self.subsystems = subsystems



    def integrate(self, x0, u):
        
        x = np.zeros((self.x_dim,1))
        for i in range(self.num_player):
            xi = x0[self.x_dims[i]] # x0: np.array, u: np.array
            ui = u[self.u_dims[i]]

            xi = self.subsystems[i].update(xi, ui)
            x[self.x_dims[i]] = xi
        
        return x
    
    def linearize(self, x0, u):

        A = []
        B = []

        for i in range(self.num_player):
            xi = x0[self.x_dims[i]]
            ui = u[self.u_dims[i]]
            Ai, Bi = self.subsystems[i].linearize(xi, ui) 

            if i == 0:
                A = Ai
                B = Bi
            else:
                A = block_diag(A,Ai)
                B = block_diag(B,Bi)
        
        return A, B
        
    def linearizeDiscrete(self, x0, u):
        Ac, Bc = self.linearize(x0, u)

        Ad = expm(Ac*self.dt)
        Bd = pinv(Ac)@(Ad- np.eye(self.x_dim))@Bc   

        return Ad, Bd
    
    def linearizeDiscrete_Interaction(self,x0,u):

        # Ad, Bd = self.linearizeDiscrete(x0, u)
        Ad, Bd = self.linearize(x0, u)

        Bds = []
        idx = 0
        for i in range(self.num_player):
            Bds.append(Bd[:, i:i+self.subsystems[i].u_dim])
            idx = idx + self.subsystems[i].u_dim

        
        return Ad, np.array(Bds)
    



class VehicleDyanmics:
    def __init__(self, x_ref, L, dt):
        self.L = L
        self.dt = dt
        self.x_dim = 4
        self.u_dim = 2
        self.x_ref = x_ref

    def dynamics(self, x, u):
        f = np.array([x[3]*np.cos(x[2]),
            x[3]*np.sin(x[2]),
            (x[3]/self.L)*np.tan(u[1]).
            u[0] ])
    
    def calc_dfdx(self, x, u):
        
        u_ = u.reshape(self.u_dim)
        x_ = x.reshape(self.x_dim)
        dfdx = np.array( [[1.0, 0.0, -x_[3]*math.sin(x_[2])* self.dt, math.cos(x_[2])* self.dt],
                          [0.0, 1.0, x_[3]*math.cos(x_[2])* self.dt, math.sin(x_[2])* self.dt],
                          [0.0, 0.0, 1.0, math.tan(u_[1])* self.dt/self.L ],
                          [0.0, 0.0, 0.0, 1.0]])

        return dfdx
    
    def calc_dfdu(self, x, u):
        u_ = u.reshape(self.u_dim)
        x_ = x.reshape(self.x_dim)

       
        dfdu = np.array([[0.0, 0.0],
                    [0.0,0.0],
                    [0.0, x_[3]* self.dt/(self.L * math.cos(u_[1])**2)],
                    [1.0*self.dt, 0.0]])

        return dfdu

    def update(self, x0, u):
        delta = u[1]
        a = u[0]

        x = x0[0] + x0[3] * np.cos(x0[2]) * self.dt
        y = x0[1] + x0[3] * np.sin(x0[2]) * self.dt
        yaw = x0[2] + x0[3]* np.tan(delta) * self.dt/ self.L 
        v = x0[3] + a * self.dt 

        # try:
        #     if abs(delta) >= 0.2:
        #         # print(yaw)
        # except:
        #     return np.array([x,y,yaw,v]).reshape(4,1)

        return np.array([float(x),float(y),float(yaw),float(v)]).reshape(4,1)
    
    def linearize(self, x, u):

        A = self.calc_dfdx(x,u)
        B = self.calc_dfdu(x,u)

        return A, B   

    


# Ego = VehicleDyanmics(2.7, 0.2)
# Human = VehicleDyanmics(2.7, 0.2)
# xR_dim = list(range(4,8))
# xH_dim = list(range(0,4))
# Interaction = InteractionDynamics(0.2, 8, 2, np.array([xR_dim,xH_dim]), [[0,1],[2,3]], 2, [Ego, Human])

# xr = [0,0,0,10]
# xh = [10,0,0,5]
# ur = [0,0]
# uh = [0,0]
# Interaction.linearizeDiscrete(np.array(xr+xh), np.array(ur+uh))