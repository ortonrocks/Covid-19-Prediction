import pandas as pd
import numpy as np
import seaborn as sns
#本次測試所使用演算法
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


df=pd.read_csv('./data_washing.csv',index_col=None)


df['test_condition'] = df['test_condition'].astype(int)

x_columns = df.columns.tolist()
x_columns = x_columns[4:-1]
len(x_columns)

for column in x_columns:
    df[column] = df[column].astype(np.float32)

df=df.drop('Unnamed: 0',axis=1)

data=df

data=data.drop(['pct_worried_finances_weighted'],axis=1)


#建立相關係數矩陣
corr = data.corr()

#如果兩個columns相關係數>0.9，刪除後者（避免共線性）
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
  for j in range(i+1, corr.shape[0]):
    if corr.iloc[i,j] >= 0.9:
      if columns[j]:
        columns[j] = False
selected_columns = data.columns[columns]
data=data[selected_columns]


#column篩選結果
selected_columns.values

print(len(selected_columns))
selected_columns[0]
data.to_csv('try.csv',encoding='utf-8-sig')
data.fillna(0)

# 刪除p值>0.05的column
import statsmodels.api as sm

selected_columns = selected_columns.values


def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns


SL = 0.05
data_modeled, selected_columns = backwardElimination(data.iloc[:, 3:-1].values, df.iloc[:, -1].values, SL,selected_columns)

#建立以最終篩選columns為基礎的data
data1 = pd.DataFrame(data = data, columns = selected_columns)

#對篩選出的columns進行確診和非確診的視覺化分析
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (20, 25))
j = 0
for i in data1.columns[:-1]:
  plt.subplot(4, 4, j+1)
  j += 1
  sns.distplot(data1[i][data['test_condition']==0], color='g', label = 'condition=0')
  sns.distplot(data1[i][data['test_condition']==1], color='r', label = 'condition=1')
  plt.legend(loc='best')
fig.suptitle('covid Data Analysis')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()



#建立X和label
X=data1.drop('test_condition',axis=1)
Y=data1['test_condition']


#train,val split
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(X,Y,test_size=0.2, random_state=42)

#LogisticRegression()
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_val)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)



# Support Vector Machines

svc = SVC()
svc.fit(x_train, y_train)
Y_pred = svc.predict(x_val)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc




#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
Y_pred = knn.predict(x_val)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)



# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
Y_pred = gaussian.predict(x_val)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)



# Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
Y_pred = perceptron.predict(x_val)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
Y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_val)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
Y_pred1 = decision_tree.predict(x_val)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(x_val)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)







models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

