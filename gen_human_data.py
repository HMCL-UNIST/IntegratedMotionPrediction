import math
import numpy as np
import bisect
import torch 
import time
import matplotlib.pyplot as plt
import matplotlib.lines as line
from scipy.stats import truncnorm
import warnings


import reference.Reference as rf
from motion_planning.MPPI import *
from motion_planning.inference import *
from motion_planning.dynamics import *
from motion_planning.vehicle_model import *
from human_model.iLQcost import *
from human_model.solveiLQgame import *
from human_model.PlayerCost import *




rd_width = 5
rd_length = 40

for trial in range(0,30000):

    ## Parameters for ILQR
    num_player = 2
    alpha_scale = 0.2
    max_iteration = 100
    tolerence = 1e-3
   
    ## Parameters for reciding horizon
    dt = 0.2
    N_ilq = 5
    N = 5
    Nd = 1
    N_sim = 60

    vH_lim = 15/3.6
    vR_lim = 15/3.6

    ## Initial State
    xH_0 = [0.0, rd_width/2, 0.0, vH_lim]
    xR_0 = [0.0, -rd_width/2, 0.0, vR_lim]


    ## State and Control limit 
    xR_lim = [0.0, rd_width, 0.0, vR_lim+0.5]
    xH_lim = [0.0, rd_width, 0.0, vH_lim]
    uR_lim = [5, 0.3]
    uH_lim = [5, 0.3]
    x_lim = np.array(xH_lim + xR_lim)
    u_lim = np.array(uH_lim + uR_lim)

    ## Generate reference trajectory
    pts_r = np.array([[xH_0[0], rd_width/2], [rd_length+30, rd_width/2]])
    pts_h = np.array([[xH_0[0], rd_width/2], [rd_length+30, rd_width/2]])
    ref_r = rf.reference(dt, 1, 1, pts_r)
    ref_h = rf.reference(dt, 1, 1, pts_h)


    ## Initionalize dynamics and optimization
    Ego = VehicleDyanmics(ref_r, 1.5, dt)
    Human = VehicleDyanmics(ref_h, 1.5, dt)
    
    xH_dim = Human.x_dim
    xR_dim = Ego.x_dim

    xH_dims = list(range(0,xH_dim))
    xR_dims = list(range(xH_dim, xR_dim+xH_dim))

    x_dim = xH_dim + xR_dim
    x_dims = np.array([xH_dims, xR_dims])

    uH_dim = Human.u_dim
    uR_dim = Ego.u_dim

    uH_dims = list(range(0,uH_dim))
    uR_dims = list(range(uH_dim,uH_dim+uR_dim))

    u_dim = uH_dim + uR_dim
    u_dims = np.array([uH_dims, uR_dims])

    dynamics_A = InteractionDynamics(dt, x_dim, u_dim, x_dims, u_dims, [Human,Ego])
    dynamics_D = InteractionDynamics(dt, xH_dim, uH_dim, x_dims, u_dims, [Human])

    Ps_A = np.zeros((num_player, N_ilq, uH_dim, x_dim))
    alpha_A = np.zeros((num_player, N_ilq, uH_dim, 1))
    Ps_D = np.zeros((1,N_ilq, 2, 4))
    alpha_D = np.zeros((1,N_ilq, 2, 1))


    '''
        Human Model
    '''

    #Attentive Human Cost
    QH = np.diag([0, 800, 80 , 10, 0.0, 0.0, 0.0, 0.0])
    RH = np.diag([0.5, 0.5])
    u_ref = np.zeros((2,1))
    Cost_Ref_A = ReferenceCost(0, ref_h, vH_lim, u_ref, x_dim, uH_dim, QH, RH) #0: Human_A, 1:Robot, 2:Human_D

    px_indices = [0,4]
    py_indices = [1,5]
    distance = 5
    Cost_Collision_A = CollisionCost(px_indices, py_indices, x_dim, uH_dim, distance)
    Cost_RoadBoundary_top = SemiQuadraticCost(x_dim, uH_dim, 1, rd_width-0.5, True)
    Cost_RoadBoundary_bottom = SemiQuadraticCost(x_dim, uH_dim, 1, 0.5, False)

    Car_H_cost_A = PlayerCost()
    Car_H_cost_A.add_cost(Cost_Ref_A, 'xu', 1e5)
    Car_H_cost_A.add_cost(Cost_Collision_A, 'x', 2e5)
    Car_H_cost_A.add_cost(Cost_RoadBoundary_top, 'x', 8e5)
    Car_H_cost_A.add_cost(Cost_RoadBoundary_bottom, 'x', 8e5)

    #Distracted Human Cost
    QH = np.diag([0, 800 , 80 , 10])
    RH = np.diag([0.5, 0.5])
    Cost_Ref_D = ReferenceCost(0, ref_h, vH_lim, u_ref, xH_dim, uH_dim, QH, RH) #0: Human, 1:Robot
    Cost_RoadBoundary_top = SemiQuadraticCost(xH_dim, uH_dim, 1, rd_width-0.5, True)
    Cost_RoadBoundary_bottom = SemiQuadraticCost(xH_dim, uH_dim, 1, 0.5, False)
    Cost_vel_max = SemiQuadraticCost(xH_dim, uH_dim, 3, 0.0, False)

    Car_H_cost_D = PlayerCost()
    Car_H_cost_D.add_cost(Cost_Ref_D, 'xu', 8e5)
    Car_H_cost_D.add_cost(Cost_RoadBoundary_top, 'x', 8e5)
    Car_H_cost_D.add_cost(Cost_RoadBoundary_bottom, 'x', 8e5)



    '''
        Robot Model
    '''

    ##Ego Vehicle Cost
    QR = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 5.0, 5.0])
    RR = np.diag([0.5, 0.5])
    RD = np.diag([0.1 ,0.1])
    Cost_Ref_R = ReferenceCost(1, ref_r, vR_lim, u_ref, x_dim, uR_dim, QR, RR) #0: Human, 1:Robot

    distance = 5
    Cost_Collision_R = CollisionCost(px_indices, py_indices, x_dim, uR_dim, distance)
    Cost_RoadBoundary_top = SemiQuadraticCost(x_dim, uR_dim, 5, rd_width, True)
    Cost_RoadBoundary_bottom = SemiQuadraticCost(x_dim, uR_dim, 5, -rd_width, False)
    Cost_steering = QuadraticCost(x_dim, uR_dim, 6, 0.0)

    Car_cost_R = PlayerCost()
    Car_cost_R.add_cost(Cost_Ref_R, 'xu', 1000)
    Car_cost_R.add_cost(Cost_Collision_R, 'x', 1000)

    Costs_A = [Car_H_cost_A, Car_cost_R]
    Costs_D = [Car_H_cost_D]


    '''
       Internal State
    '''

    ## Initialize the probability distribution
    beta_lim = [0.2, 1.0]
    beta_distr = beta_prob_distr(0.2, beta_lim)
    theta_prob = [0.5, 0.5]
    beta = np.random.rand(1)*0.8 + 0.2
    if beta <= beta_lim[0]:
        beta = beta_lim[0]
    elif beta >= beta_lim[1]:
        beta = beta_lim[1]
    # beta = 0.2

    if trial < 15000:
        theta = 'a'
    else:
        theta = 'd'

    '''
        Initialization of human decision-making and robot optimization
    '''
    
    x0 = np.array(xH_0 + xR_0)
    u0 = np.zeros((4,1))
    ilq_results_A = ilq_results(xHdims, dynamics_A, Ps_A, \
                            alpha_A, Costs_A, uH_dim, u_lim, \
                            alpha_scale, N_ilq, max_iteration, tolerence)
    
    ilq_results_D = ilq_results(xHdims, dynamics_D, Ps_D,\
                            alpha_D, Costs_D, uH_dim, u_lim, \
                            alpha_scale, N_ilq, max_iteration, tolerence)

    
    W = 700   #weight for collision
    K = 1000  #Number of samples for autonomous vehicle motion
    K2 = 2    #Number of samples for human motion
    ca_lat_bd = 2.0
    ca_lon_bd = 4

    Ego_opt = mppi(N, K, K2, x_lim, u_lim, QR, RR, RD, W, wac = 1, ca_lat_bd=ca_lat_bd, ca_lon_bd=ca_lon_bd)


    Ego_traj = []
    Human_traj = []

    Ego_input = []
    Human_input = []

    LaneChange = False
    Collision = False
    Collision2wall = False
    Collision_arr = []


    for t in range(N_sim):
        
        print("-----------------------------------------------------")
        print("trial", trial, "simuation time:", t, "true beta", beta)

        start = time.time()
        
        '''
        Planning Robot Action
        '''

        ilq_results_A.solveiLQgame(x0)
        ilq_results_D.solveiLQgame(x0[:4])


        traj_r, states, uR_arr, xH, num_sampled_A = Ego_opt.solve_mppi(x0, RH, ilq_results_A, ilq_results_D, theta_prob, beta_distr, t=None, Human_seq=None, learning=False)
        uR = uR_arr[0]
        xR = Ego.update(x0[4:],uR)

        
        '''
        Update Human action
        '''
        uH_A = ilq_results_A.ilq_solve.best_operating_point.us[0,:2]
        uH_D = ilq_results_D.ilq_solve.best_operating_point.us[0,:2]

        ## Attentive Human
        u0_A = np.array([float(uH_A[0]), float(uH_A[1]),uR[0], uR[1]])
        A, B = dynamics_A.linearizeDiscrete_Interaction(x0, u0_A)
        Sigma_A = get_covariance(RH, B[0], ilq_results_A.ilq_solve.best_operating_point.Zs[0,0])
        Sigma_A = np.linalg.inv(Sigma_A)
        Sigma_A = np.abs(Sigma_A)

        ## Distracted Human
        u0_D = np.array([float(uH_D[0]), float(uH_D[1]),uR[0], uR[1]])
        A, B = dynamics_D.linearizeDiscrete_Interaction(x0, u0_D)
        Sigma_D = get_covariance(RH, B[0], ilq_results_D.ilq_solve.best_operating_point.Zs[0,0])
        Sigma_D = np.linalg.inv(Sigma_D)
        Sigma_D = np.abs(Sigma_D)

        if theta == 'a':
            uH_a = np.random.normal(u0_A[0],  Sigma_A[0,0]/beta, size =1)
            uH_delta = np.random.normal(u0_A[1], Sigma_A[1,1]/beta, size =1)
        
            u0[0] = np.clip(uH_a,-uH_lim[0], uH_lim[0])
            u0[1] = np.clip(uH_delta,-uH_lim[1], uH_lim[1]) 
        elif theta == 'd':
            uH_a = np.random.normal(u0_D[0],  Sigma_D[0,0]/beta, size =1)
            uH_delta = np.random.normal(u0_D[1], Sigma_D[1,1]/beta, size =1)
     
            u0[0] = np.clip(uH_a,-uH_lim[0], uH_lim[0])
            u0[1] = np.clip(uH_delta,-uH_lim[1], uH_lim[1])
        else:
            print("Human model is WRONG")
            break
       
        xH = Human.update(x0[:4],u0[:2])
            
        


        '''
        Update Belief over Human Internal State
        '''
        theta_, prob_beta_a, prob_beta_d  = updateTheta(theta_prob, beta_distr, u0[:2].reshape(2,), \
                        uH_A.reshape(2,), uH_D.reshape(2,), Sigma_A, Sigma_D)

        beta_distr.prob_a = prob_beta_a
        beta_distr.prob_d = prob_beta_d
        beta_ = np.argmax(prob_beta_a)
        if beta_ == 0:
            beta_distr.beta_a = 0.2
        else:
            beta_distr.beta_a = 1

        beta_ = np.argmax(prob_beta_d)
        if beta_ == 0:
            beta_distr.beta_d= 0.2
        else:
            beta_distr.beta_d = 1
        
        if np.isnan(theta_prob[0]) or np.isnan(theta_prob[1]):
            print(theta_)       
        else:
            theta_prob = theta_
        

        '''
            Print the results
        '''            
        if theta == 'a':
            print("predicted beta is: %f" %(beta_distr.beta_a))
            error = round( math.sqrt((beta - beta_distr.beta_a)**2),2)
            print("error truncated beta is: %f" %(error))
            print("curent_prob: ", theta_prob[0])
            
        elif theta == 'd':
            print("predicted beta is: %f" %(beta_distr.beta_d))
            error = round( math.sqrt((beta - beta_distr.beta_d)**2),2)
            print("error truncated beta is: %f" %(error))
            print("curent_prob: ", theta_prob[1])
           
        if xH[0] >= xH_lim[1]/2 and xH[1] >= xH_lim[1]:
            xH[1] = xH_lim[1]
        elif xH[0] >= xH_lim[1]/2 and xH[1] <= 0:
            xH[1] = 0
        if xH[3] >= xH_lim[3] +1:
            xH[3] = xH_lim[3] +1
        x0[:4] = xH.reshape(4,)
        x0[4:] = xR.reshape(4,)


        '''
            Check the collision
        '''
        
        psi = xH[2]
        dist_x =  (xH[0] - xR[0])*math.cos(psi) + (xH[1] - xR[1])*math.sin(psi)
        dist_y =  (xH[0] - xR[0])*math.sin(psi) - (xH[1] - xR[1])*math.cos(psi)
        cl = ((dist_x**2)/((ca_lon_bd)**2) + (dist_y**2)/((ca_lat_bd)**2))


        if cl <= 1 or Collision:
            print("Collision to Vehicle!!!!!!!")
            Collision = True

        
        if xR[1] > rd_width or xR[1] < -rd_width:
            Collision = True
            print("Collision to WAll!!!!!!!")

        if not Collision and abs(xR[1]-rd_width/2) < 0.5 :
            LaneChange = True
            print("Success Lane change")

        ## For saving data    
        x_ = [float(xH[0]), float(xH[1]), float(xH[2]), float(xH[3]), float(xR[0]), float(xR[1]), float(xR[2]), float(xR[3])]
        Ego_traj.append(x_[4:])
        Human_traj.append(x_[:4])

        Ego_input.append([float(uR[0]), float(uR[1])])
        Human_input.append([float(u0[0]), float(u0[1])])
        
        print("-----------------------------------------------------")

    print("Simulation Done")
        
    
    if theta == 'a':
        file_name = './train_inference/train_data/attentive'+str(trial)
    else:
        file_name = './train_inference/train_data/distracted'+str(trial)

    np.savez_compressed(file_name, \
                        ego_traj = Ego_traj, \
                        human_traj = Human_traj, \
                        ego_input = Ego_input,\
                        human_input = Human_input,\
                        t_beta = beta, \
                        t_theta = theta, \
                        LaneChange = LaneChange,\
                        Collision = Collision )    

    
    


