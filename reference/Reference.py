import math
import numpy as np
import bisect
import matplotlib.pyplot as plt
import reference.cubic_spline as cs

class reference:
    def __init__(self, dt, d_dist, ds, pts):
        self.ds = ds
        self.ref_pts = pts
        self.generate_traj()
        self.dt = dt
        self.d_dist = d_dist
    
    def generate_traj(self): 

        traj_x = []
        traj_y = []
        traj_psi = []
        pass_id = []
        for i in range(len(self.ref_pts)-1):

            if i in pass_id:
                continue

            if self.ref_pts[i+1][0] == self.ref_pts[i][0] or self.ref_pts[i+1][1] == self.ref_pts[i][1]: # in same line
                x = [self.ref_pts[i][0], self.ref_pts[i+1][0]]
                y = [self.ref_pts[i][1], self.ref_pts[i+1][1]]
                cx, cy, cyaw, ck, s = cs.calc_spline_course(x, y, ds=self.ds)
                traj_x = np.append(traj_x, cx)
                traj_y = np.append(traj_y, cy)
                traj_psi = np.append(traj_psi, cyaw)
                pass_id = []
            else:
                x = [self.ref_pts[i][0], self.ref_pts[i+1][0], self.ref_pts[i+2][0]]
                y = [self.ref_pts[i][1], self.ref_pts[i+1][1], self.ref_pts[i+2][1]]
                cx, cy, cyaw, ck, s = cs.calc_spline_course(x, y, ds=self.ds)
                traj_x = np.append(traj_x, cx)
                traj_y = np.append(traj_y, cy)
                traj_psi = np.append(traj_psi, cyaw)
                pass_id = [i+1]
        
        self.cx = traj_x
        self.cy = traj_y
        self.cyaw = traj_psi
        self.length = len(cx)
    
    def nearest_index(self, x0, pred = False):


        dx = [x0[0] - x for x in self.cx]
        dy = [x0[1] - y for y in self.cy]
        dist = np.hypot(dx, dy)
        ind_in_N = int(np.argmin(dist))
        ind = ind_in_N

        if not pred:
            self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(x0[2] + math.pi / 2.0)],
                                         [math.sin(x0[2] + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er

    def calc_ref_trajectory_in_T_step(self, x0, N, ref_speed, pred=False):
        z_ref = np.zeros((N+1, len(x0)))
        length = self.length
        ind, _ = self.nearest_index(x0, pred)

        # ind = ind
        z_ref[0, 0] = self.cx[ind]
        z_ref[0, 1] = self.cy[ind]
        z_ref[0, 2] = self.cyaw[ind]
        z_ref[0, 3] = ref_speed

        dist_move = 0.0

        for i in range(1, N + 1):
            dist_move += abs(x0[3]) * self.dt
            try:
                ind_move = int(round(dist_move / self.d_dist))
            except:
                ind_move = int(np.round(dist_move / self.d_dist))
            index = min(ind + ind_move, length - 1)

            z_ref[i, 0] = self.cx[index]
            z_ref[i, 1] = self.cy[index]
            z_ref[i, 2] = self.cyaw[index]
            z_ref[i, 3] = ref_speed
        return z_ref, ind
            

if __name__ == '__main__':
    pts = np.array([[0.0, 0.0], [20.0, 0.0]])
    ref = reference(0.5,pts)

    plt.plot(self.ref_pts[:,0], self.ref_pts[:,1], "xb", label="input")
    plt.plot(traj_x, traj_y, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.show()