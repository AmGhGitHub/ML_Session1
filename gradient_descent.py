import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gradient_descent(x, y):
    m_curr = b_curr = 0
    max_iter = 100
    n = float(len(x))
    lr = 0.001
    vals = np.zeros((max_iter, 4))
    for _iter in range(max_iter):
        y_predict = m_curr * x + b_curr
        md = -(2 / n) * np.sum(x * (y + -y_predict))
        bd = -(2 / n) * np.sum(y - y_predict)
        m_curr -= lr * md
        b_curr -= lr * bd
        cost_val = np.sum(np.square(y - y_predict)) / n
        vals[_iter][0] = _iter
        vals[_iter][1] = m_curr
        vals[_iter][2] = b_curr
        vals[_iter][3] = cost_val
        # print(f'Iter:{i}-->m={m_curr:.4f} b={b_curr:.4f} cost={cost_val:.4f}')

    df = pd.DataFrame(data=vals, columns=['iter', 'm', 'b', 'mse'])
    df.set_index(['iter'], inplace=True)
    return df


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])
    df_m_b_mse = gradient_descent(x, y)
    fig, ax = plt.subplots(nrows=1, ncols=3)

    for i, ax_ in enumerate(ax):
        df_m_b_mse[df_m_b_mse.columns[i]].plot(ax=ax_)
        ax_.set_xlabel('Iter #')
        ax_.set_ylabel(df_m_b_mse.columns[i])

    plt.tight_layout()
    plt.show()
