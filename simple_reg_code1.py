import matplotlib.pyplot as plt
import numpy as np


def convert_lis_to_2darray(input_list):
    return np.array(input_list).reshape(-1, 1)


actual_capex = [[46], [48], [50], [54], [58.5]]
actual_npv = [70, 90.2, 130, 175.1, 180]

plot_training_data = False
if plot_training_data:
    fig, ax = plt.subplots()
    ax.scatter(actual_capex, actual_npv,
               marker='s', s=50, c='#fa574b',
               label='Train Data')
    font_dict = {'fontsize': 18, 'fontweight': 'bold'}
    ax.set_title("NPV vs CAPEX", **font_dict)
    ax.set_xlabel("CAPEX, MM$")
    ax.set_ylabel("NPV, MM$")
    ax.legend(loc='lower right')
    ax.set_xlim((40, 60))
    ax.set_ylim((40, 200))
    ax.grid(True)

completion_time = [3, 2, 1, 3, 1]
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(actual_capex, actual_npv)
predicted_npv = model.predict([[52]])
print(f"CAPEX of $MM 52 should bring about an NPV of $MM {predicted_npv[0]:.1f}")

# fitness of the model
plot_fitness = False
if plot_fitness:
    fig, ax = plt.subplots()

    ax.set_title("NPV vs CAPEX",
                 fontdict={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel("CAPEX, MM$")
    ax.set_ylabel("NPV, MM$")
    ax.set_xlim((40, 60))
    ax.set_ylim((40, 200))
    ax.plot([40, 60], [127, 127], 'g:', lw=3, label='1st ft-line')
    ax.plot([45, 57], [40, 200], c='#4b88fa', ls='-', lw=3, label='2nd ft-line')
    ax.plot([42, 59], [40, 200], 'k--', lw=2, label='3rd ft-line')
    ax.scatter(actual_capex, actual_npv,
               marker='s', s=50, c='#fa574b',
               label='Train Data')
    ax.legend(loc='lower right')
    ax.grid(True)

plot_residual = True
if plot_residual:
    # calculate the residuals
    residual_npv = []
    for (capex, npv) in zip(actual_capex, actual_npv):
        residual_npv.append(model.predict([capex])[0] - npv)

    fig, ax = plt.subplots()

    ax.set_title("NPV vs CAPEX",
                 fontdict={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel("CAPEX, MM$")
    ax.set_ylabel("NPV, MM$")
    ax.set_xlim((40, 60))
    ax.set_ylim((40, 200))
    ax.scatter(actual_capex, actual_npv,
               marker='s', s=50, c='#fa574b',
               label='Train Data')

    ax.plot([40, 60], [model.predict([[40]])[0], model.predict([[60]])[0]], 'g:', lw=3, label='Line of Best fit')
    ax.vlines(actual_capex, actual_npv, np.array(residual_npv) + np.array(actual_npv), label='Residual')

    ax.legend(loc='lower right')
    ax.grid(True)

plt.show()
