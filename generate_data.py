import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class GenerateData:
    def __init__(self, x_min, x_max, n_points, slope, intercept, noisy=True):
        self.__noise = np.zeros((1, n_points))
        if noisy:
            self.__noise = np.random.randn(n_points)
        self.x = np.linspace(x_min, x_max, n_points).reshape(1, -1)
        self.y = slope * self.x + intercept + self.__noise
        self.slope = slope
        self.intercept = intercept

    def plot(self, plot_line=False):
        fig, ax = plt.subplots()
        ax.scatter(self.x, self.y, label='data points', c='red')
        if plot_line:
            ax.plot(self.x, self.slope * self.x + self.intercept, label=f'y={self.slope:.1f}x+{self.intercept:.1f}')
            ax.vlines(self.x, self.y, self.y - self.__noise, label='error')
        ax.legend(loc=0)
        font_dict = dict(size=16, weight='bold')
        ax.set_xlabel('x', fontdict=font_dict)
        ax.set_ylabel('y', fontdict=font_dict)
        ax.set_title('Data used in building MSE function for SIMPLE LINEAR REGRESSION')
        return fig
