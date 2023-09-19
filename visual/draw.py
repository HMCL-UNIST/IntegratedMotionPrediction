import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

class Arrow:
    def __init__(self, x, y, theta, L, c):
        angle = np.deg2rad(30)
        d = 0.4 * L
        w = 2

        x_start = x
        y_start = y
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)

        theta_hat_L = theta + math.pi - angle
        theta_hat_R = theta + math.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)


def draw_car(x, y, yaw, steer, C, Collision, alpha, color='white',):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])

    rrWheel += np.array([[-C.WB], [-C.WD / 2]])
    rlWheel += np.array([[-C.WB], [C.WD / 2]])

    # rrWheel[1, :] -= C.WD / 2
    # rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    if Collision:
        plt.fill(car[0, :], car[1, :], color='r', alpha=0.8)
    # else:
    #     plt.fill(car[0, :], car[1, :], color=color, alpha=0.2)

    elif color !='white':
        plt.fill(car[0, :], car[1, :], color=color, alpha=alpha)

    plt.plot(car[0, :], car[1, :], 'k')
    plt.plot(frWheel[0, :], frWheel[1, :], 'k')
    plt.plot(rrWheel[0, :], rrWheel[1, :], 'k')
    plt.plot(flWheel[0, :], flWheel[1, :], 'k')
    plt.plot(rlWheel[0, :], rlWheel[1, :], 'k')
    plt.fill(frWheel[0, :], frWheel[1, :], 'k')
    plt.fill(rrWheel[0, :], rrWheel[1, :], 'k')
    plt.fill(flWheel[0, :], flWheel[1, :], 'k')
    plt.fill(rlWheel[0, :], rlWheel[1, :], 'k')
    
    Arrow(x, y, yaw, C.WB * 0.6, 'k')
