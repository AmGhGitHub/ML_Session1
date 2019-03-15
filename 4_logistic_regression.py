import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics

diabetes = pd.read_csv('data/diabetes.csv', header=[0])
desired_columns = ['pregnancy', 'insulin', 'bmi', 'age', 'outcome']
X, y = diabetes[desired_columns[:-1]], diabetes[desired_columns[-1]]
# Check whether stratification is required or not ##
# sns.countplot(y)
# print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
lgr = LogisticRegression()
lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)
print(f'Model accuracy score for the test data set is: {metrics.accuracy_score(y_test, y_pred):.4f}')
# print('True:', y_test.values[:25])
# print('Pred:', y_pred[:25])

cfm = metrics.confusion_matrix(y_test, y_pred)
# print(cfm)

TP = cfm[1, 1]
TN = cfm[0, 0]
FP = cfm[0, 1]
FN = cfm[1, 0]
# sns.heatmap(cfm, cmap='plasma', annot=True, annot_kws={'size': 16}, fmt='g')
# plt.ylabel('True Class')
# plt.xlabel('Predicted Class')
# plt.show()

# classification accuracy
cls_accuracy = (TP + TN) / (TP + TN + FP + FN)
# print(f'Classification accuracy is: {cls_accuracy:.4f}, {metrics.accuracy_score(y_test, y_pred)}')

# mis-classification accuracy
# print(1 - cls_accuracy)
sensitivity = TP / (TP + FN)
# print(f'Classification sensitivity or recall is: {sensitivity:.4f}, {metrics.recall_score(y_test, y_pred):.4f}')

specificity = TN / (TN + FP)
# print(f'Classification specificity is: {specificity:.4f}')

# based on the obtained values, our classifier is highly specified but not sensitive

# False positive rate
falsepositive_rate = 1 - specificity

print('Actual:', y_test[:10].values)
print('Prediction: ', lgr.predict(X_test)[:10])
# print the first 10 predicted probabilities for class membership. For example it is important to call a person with P>80 immediately
print("   Class:0      Class:1  ")
print(lgr.predict_proba(X_test)[:10])
