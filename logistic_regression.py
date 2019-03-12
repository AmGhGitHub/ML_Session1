import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

diabetes = pd.read_csv('data/diabetes.csv', header=[0])
desired_columns = ['pregnancy', 'insulin', 'bmi', 'age', 'outcome']
X, y = diabetes[desired_columns[:-1]], diabetes[desired_columns[-1]]
# Check whether stratification is required or not ##
# sns.countplot(y)
# print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
a=1


plt.show()
