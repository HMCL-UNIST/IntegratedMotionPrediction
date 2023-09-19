import numpy as np
import math
import torch
from scipy.stats import norm


class beta_prob_distr:
    def __init__(self, beta_init, beta_lim):
        self.beta_a = beta_init
        self.beta_d = beta_init
        self.prob_a = [1/2, 1/2]
        self.prob_d = [1/2, 1/2]
        self.beta_lim = beta_lim


def updateTheta(theta_now, beta_distr_now, uH, uH_A, uH_D, u_A_sigma, u_D_sigma):
    
    x_next_sig_a1 = u_A_sigma/ 0.2
    x_next_sig_a2 = u_A_sigma

    x_next_sig_d1 = u_D_sigma/ 0.2
    x_next_sig_d2 = u_D_sigma

    pdf_a1 = calc_pdf(uH, uH_A, x_next_sig_a1)
    pdf_a2 = calc_pdf(uH, uH_A, x_next_sig_a2)

    pdf_d1 = calc_pdf(uH, uH_D, x_next_sig_d1)
    pdf_d2 = calc_pdf(uH, uH_D, x_next_sig_d2)


    pdf_a_prev = beta_distr_now.prob_a
    pdf_d_prev = beta_distr_now.prob_d

    ## Attentive
    prob_a1 = pdf_a1*pdf_a_prev[0]/(pdf_a1*pdf_a_prev[0] + pdf_a2*pdf_a_prev[1])
    prob_a2 = pdf_a2*pdf_a_prev[1]/(pdf_a1*pdf_a_prev[0] + pdf_a2*pdf_a_prev[1])

    ## Distracted
    prob_d1 = pdf_d1*pdf_d_prev[0]/(pdf_d1*pdf_d_prev[0] + pdf_d2*pdf_d_prev[1])
    prob_d2 = pdf_d2*pdf_d_prev[1]/(pdf_d1*pdf_d_prev[0] + pdf_d2*pdf_d_prev[1])

  
    denom = theta_now[0]*(pdf_a1*pdf_a_prev[0] + pdf_a2*pdf_a_prev[1])
    denom2 = theta_now[1]*(pdf_d1*pdf_d_prev[0] + pdf_d2*pdf_d_prev[1])

    theta_ = denom/(denom+denom2)
    theta = [theta_, 1-theta_]


    return theta, [prob_a1, prob_a2], [prob_d1, prob_d2]    


def calc_pdf(u, u_pred, u_sigma):
    pdf_u_acc = norm(u_pred[0], u_sigma[0,0]).pdf(u[0])
    pdf_u_del = norm(u_pred[1], u_sigma[1,1]).pdf(u[1])
    pdf = (pdf_u_acc + pdf_u_del)/2

    return pdf