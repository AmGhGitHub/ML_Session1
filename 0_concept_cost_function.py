'''
Author: Amir Ghaderi
Date: 15 March 2019
Demonstration of cost function definition (mean square error) and application
'''

import numpy as np
import matplotlib.pyplot as plt
import generate_data as gd
from mpl_toolkits.mplot3d import Axes3D


def cost_function(data_points, beta0, beta1):
    """
    :param data_points: actual data object which actual contain (x,y) values
    :param beta0: intercept value in simple linear regression
    :param beta1: slope value in y=beta0+beat1*x
    :return: mean square error
    """
    x = data_points.x
    y_true = data_points.y
    y_cal = beta0 + beta1 * x
    m = len(x)
    mse = (1.0 / (2 * m)) * np.sum(np.square(y_cal - y_true))
    return mse


def calc_plot_mse(data, beta0_min, beta0_max, beta1_min, beta1_max, n_divisions):
    beta0_rng = np.linspace(beta0_min, beta0_max, n_divisions)
    beta1_rng = np.linspace(beta1_min, beta1_max, n_divisions)
    beta0, beta1 = np.meshgrid(beta0_rng, beta1_rng)
    mse_val = np.array(
        [cost_function(data_points=data, beta0=beta0_, beta1=beta1_)
         for (beta0_, beta1_) in
         zip(beta1.flatten(), beta0.flatten())])

    mse_val = mse_val.reshape(beta0.shape)

    fig = plt.figure()
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')  # MSE surface plot
    ax1 = fig.add_subplot(1, 2, 2)  # MSE contour plot

    mse_surface = ax0.plot_surface(beta0, beta1, mse_val, cmap='viridis')
    plt.colorbar(mse_surface, ax=ax0)
    ax0.set_xlabel('\u03B20')
    ax0.set_ylabel('\u03B21')
    ax0.set_zlabel('MSE')
    ax0.set_title('3D Plot of MSE as a function of \u03B20 & \u03B21')

    mse_contour = ax1.contour(beta0, beta1, mse_val, levels=[1, 10, 50, 100, 200, 500, 1000], colors='black')
    ax1.clabel(mse_contour, inline=1, fontsize=10)
    ax1.set_xlabel('\u03B20')
    ax1.set_ylabel('\u03B21')
    ax1.set_title('Contour Plot of MSE as a function of \u03B20 & \u03B21')

    return mse_val, fig


if __name__ == '__main__':
    xy_points = gd.GenerateData(
        x_min=0, x_max=5, n_points=11,
        slope=2.5, intercept=1.8,
        noisy=False)

    xy_points.plot(False)

    cost_val, figure = calc_plot_mse(
        data=xy_points,
        beta0_min=-10, beta0_max=10,
        beta1_min=-10, beta1_max=10,
        n_divisions=100)

    plt.tight_layout()
    plt.show()
