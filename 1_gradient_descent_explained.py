import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import generate_data as gd
from sklearn.linear_model import LinearRegression


def gradient_descent(x, y):
    beta0 = beta1 = 0
    max_iter = 1000000
    # play with learning rate and see how impactful it is on the convergence speed
    # lr = 0.01
    lr = 0.01
    parameters_values = np.zeros((max_iter + 1, 3))
    iter_number = 0
    max_mse = 1e-5
    mse = 1.0
    m = len(x)
    while iter_number <= max_iter and mse >= max_mse:
        y_predict = beta0 + beta1 * x
        mse = np.sum(np.square(y - y_predict)) / m
        parameters_values[iter_number][0] = beta0
        parameters_values[iter_number][1] = beta1
        parameters_values[iter_number][2] = mse
        der_beta1 = -(2 / m) * np.sum(x * (y + -y_predict))
        der_beta0 = -(2 / m) * np.sum(y - y_predict)
        beta1 = beta1 - lr * der_beta1
        beta0 = beta0 - lr * der_beta0
        iter_number += 1

    df = pd.DataFrame(data=parameters_values[:iter_number, :], columns=['\u03B20', '\u03B21', 'mse'])
    return df


if __name__ == '__main__':
    data_points = gd.GenerateData(
        x_min=0, x_max=5, n_points=10,
        slope=-2, intercept=3,
        noisy=False)

    data_points.plot()

    x_val = data_points.x
    y_val = data_points.y
    print(np.vstack((x_val, y_val)))

    df_m_b_mse = gradient_descent(x_val, y_val)
    print(f'Iter={df_m_b_mse.index[-1]:,} \u03B21={df_m_b_mse.iloc[-1, 1]:.2f} \u03B20={df_m_b_mse.iloc[-1, 0]:.2f}')
    lr = LinearRegression().fit(x_val[:, np.newaxis], y_val[:, np.newaxis])
    print(f'ML: \u03B21={lr.coef_[0, 0]:.2f} \u03B20={lr.intercept_[0]:.2f}')
    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.tight_layout()
    font_dict = dict(size=16, weight='bold')
    for i, (ax_, c, ls, lw) in enumerate(zip(ax, ['#84d127', '#84d127', '#d18a26'], ['--', '--', '-'], [3] * 3)):
        df_m_b_mse[df_m_b_mse.columns[i]].plot(ax=ax_, lw=lw, c=c, ls=ls)
        if i == 2:
            ax_.set_yscale('log')
        ax_.set_xlabel('iterations', fontdict=font_dict)
        ax_.set_ylabel(df_m_b_mse.columns[i], fontdict=font_dict)

        tick_font_size = 14
        for tick in ax_.xaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font_size)
        for tick in ax_.yaxis.get_major_ticks():
            tick.label.set_fontsize(tick_font_size)

    plt.show()
