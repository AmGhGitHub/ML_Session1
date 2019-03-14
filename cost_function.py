import numpy as np
import matplotlib.pyplot as plt
import generate_data as gd
from mpl_toolkits.mplot3d import Axes3D


def calc_mse(data, beta0, beta1):
    x = data.x
    y_true = data.y
    y_cal = beta0 + beta1 * x
    m = len(x)
    mse = (1.0 / (2 * m)) * np.sum(np.square(y_cal - y_true))
    return mse


if __name__ == '__main__':
    data = gd.GenerateData(0, 5, 11, 2.5, 1.8)
    data.plot()

    n_div = 200
    beta0_rng = np.linspace(-100, 100, n_div)
    beta1_rng = np.linspace(-100, 100, n_div)
    beta0, beta1 = np.meshgrid(beta0_rng, beta1_rng)
    mse_val = np.array(
        [calc_mse(data=data, beta0=beta0_, beta1=beta1_) for (beta0_, beta1_) in
         zip(beta1.flatten(), beta0.flatten())])

    mse_val = mse_val.reshape(beta0.shape)

    fig = plt.figure(figsize=(16, 8))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2)

    mse_surface = ax0.plot_surface(beta0, beta1, mse_val, cmap='viridis')
    plt.colorbar(mse_surface, ax=ax0)
    ax0.set_xlabel('\u03B20')
    ax0.set_ylabel('\u03B21')
    ax0.set_zlabel('MSE')
    ax0.set_title('3D Plot of MSE as a function of \u03B20 & \u03B21')

    mse_contour = ax1.contour(beta0, beta1, mse_val, levels=[10, 50, 100, 200, 500, 1000], colors='black')
    ax1.clabel(mse_contour, inline=1, fontsize=10)
    ax1.set_xlabel('\u03B20')
    ax1.set_ylabel('\u03B21')
    ax1.set_title('Contour Plot of MSE as a function of \u03B20 & \u03B21')

    plt.show()
