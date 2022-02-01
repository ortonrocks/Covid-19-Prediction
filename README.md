# Covid-19-Prediction

Covid-19 美國各周預測

利用現成資料集對美國各州COVID進行預測
利用 LogisticRegression, SVC, LinearSVC, RandomForestClassifier,KNeighbors
Classifier,GaussianNB, Perceptron,SGDClassifier,DecisionTreeClassifier等演算法預測，

*最終利用DNN作為解決方案，設計二分類模型
------------------
model=models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(92,)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(layers.Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(layers.Dense(16,activation='relu'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(layers.Dense(8,activation='relu'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
---------------------
將準確率提高至0.655
