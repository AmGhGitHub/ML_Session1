import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gradient_descent(x, y):
    m = b = 0
    max_iter = 100000
    lr = 0.001
    parameters_values = np.zeros((max_iter, 3))
    iter_number = 0
    max_mse = 1e-4
    mse = 1.0
    n = float(len(x))
    while iter_number < max_iter and mse > max_mse:
        y_predict = m * x + b
        md = -(2 / n) * np.sum(x * (y + -y_predict))
        bd = -(2 / n) * np.sum(y - y_predict)
        m -= lr * md
        b -= lr * bd
        mse = np.sum(np.square(y - y_predict)) / n
        parameters_values[iter_number][0] = m
        parameters_values[iter_number][1] = b
        parameters_values[iter_number][2] = mse
        iter_number += 1

    df = pd.DataFrame(data=parameters_values[:iter_number, :], columns=['m', 'b', 'mse'])
    return df


if __name__ == '__main__':
    x_val = np.array([1, 2, 3, 4, 5])
    y_val = np.array([5, 7, 9, 11, 13])
    df_m_b_mse = gradient_descent(x_val, y_val)

    fig, ax = plt.subplots(nrows=1, ncols=3)

    for i, ax_ in enumerate(ax):
        df_m_b_mse[df_m_b_mse.columns[i]].plot(ax=ax_)
        ax_.set_xlabel('Iter #')
        ax_.set_ylabel(df_m_b_mse.columns[i])

    plt.tight_layout()
    plt.show()
