import math
import numpy as np

class Vehicle:

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 10.0 / 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.2  # time step
    d_dist = 1.0  # dist step

    # vehicle config
    RF = 1.4  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.4  # [m] distance from rear to vehicle back end of vehicle
    W = 1.8  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 1.0  # [m] Wheel base
    TR = 0.35  # [m] Tyre radius
    TW = 0.5  # [m] Tyre width


    def __init__(self, x0, xf, x_lim, u_lim, ref, N, id):
        self.Nx = len(x0)
        self.Nu = len(u_lim)
        self.N = N
        
        self.x0 = x0
        self.xf = xf
        self.x = self.x0
        self.xlim = x_lim

        self.ulim = u_lim
        self.u = self.radom_input()

        self.traj = self.calc_trajectory(self.x, self.u)

        self.ref_path = ref
        self.id = id  
        self.time = 0.0

        self.u_sig = None

    
    def limit_input_delta(self, delta):
        if delta >= self.ulim[1]:
            return self.ulim[1]

        if delta <= -self.ulim[1]:
            return -self.ulim[1]
        return delta

    def limit_input_acc(self,a):
        if a >= self.ulim[0]:
            return self.ulim[0]

        if a <= -self.ulim[0]:
            return -self.ulim[0]
        return a

    def radom_input(self):
        return np.zeros((self.N, self.Nu))

    def update(self, a, delta):
        delta = self.limit_input_delta(delta)
        a = self.limit_input_acc(a)

        x = self.x[0] + self.x[3] * math.cos(self.x[2]) * Vehicle.dt
        y = self.x[1] + self.x[3] * math.sin(self.x[2]) * Vehicle.dt
        yaw = self.x[2] + self.x[3]* math.tan(delta) * Vehicle.dt / Vehicle.WB 
        v = self.x[3] + a * Vehicle.dt

        self.x = [x, y, yaw, v]
    
    def calc_trajectory(self, x0, u):

        traj = np.zeros((self.N+1, self.Nx))
        traj[0,:] = np.array(x0)

        for i in range(self.N):
            delta = self.limit_input_delta(u[i,1])
            a = self.limit_input_acc(u[i,0])

            traj[i+1,0] = traj[i,0] + traj[i,3] * math.cos(traj[i,2]) * Vehicle.dt
            traj[i+1,1] = traj[i,1] + traj[i,3] * math.sin(traj[i,2]) * Vehicle.dt
            traj[i+1,2] = traj[i,2] + traj[i,3] / Vehicle.WB * math.tan(delta) * Vehicle.dt
            traj[i+1,3] = traj[i,3] +  a * Vehicle.dt

        return traj

    def calc_linear_discrete_model(self, x0, u):
        A = np.array([[1.0, 0.0, - Vehicle.dt * x0[3] * math.sin(x0[2]), Vehicle.dt * math.cos(x0[2])],
                    [0.0, 1.0, Vehicle.dt * x0[3] * math.cos(x0[2]), Vehicle.dt * math.sin(x0[2]) ],
                    [0.0, 0.0, 1.0, Vehicle.dt * math.tan(u[1]) / Vehicle.WB],
                    [0.0, 0.0, 0.0, 1.0]])

        B = np.array([[0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, Vehicle.dt * x0[3] / (Vehicle.WB * math.cos(u[1]) ** 2)],
                    [Vehicle.dt, 0.0]])

        C = np.array([Vehicle.dt * x0[3] * math.sin(x0[2]) * x0[2],
                    -Vehicle.dt * x0[3] * math.cos(x0[2]) * x0[2],
                    -Vehicle.dt * x0[3] * u[1] / (Vehicle.WB * math.cos(u[1]) ** 2),
                    0.0])

        return A, B, C    