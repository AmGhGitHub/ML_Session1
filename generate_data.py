"""
Author: Amir Ghaderi
Date: 15 March 2019
This class is aim at generation quasi-random value pairs
using straight line approach
"""

import numpy as np
import matplotlib.pyplot as plt


class GenerateData:

    def __init__(self, x_min, x_max, n_points, slope, intercept, noisy=True):
        '''
        :param x_min:
        :param x_max:
        :param n_points:
        :param slope:
        :param intercept:
        :param noisy:
        '''
        np.random.seed(42)
        self.__noise = np.zeros((1, n_points))
        if noisy:
            self.__noise = np.random.randn(n_points)
        self.x = np.linspace(x_min, x_max, n_points).flatten()
        self.y = (slope * self.x + intercept + self.__noise).flatten()
        self.slope = slope
        self.intercept = intercept

    def plot(self, plot_fit=True):
        '''
        :param plot_fit: True or False
        :return: the data canvas
        '''
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(self.x, self.y, 'ro', label='data points')
        if plot_fit:
            ax.plot(self.x, self.slope * self.x + self.intercept, label=f'y={self.slope:.2f}x+{self.intercept:.2f}',
                    color='blue')
            ax.vlines(self.x, self.y, self.y - self.__noise, label='error')

        ax.legend(loc=0)
        font_dict = dict(size=16, weight='bold')
        ax.set_xlabel('x', fontdict=font_dict)
        ax.set_ylabel('y', fontdict=font_dict)
        ax.set_title('Generated Data Points - Fitted Line - Errors')
        fig.tight_layout()
        return fig
