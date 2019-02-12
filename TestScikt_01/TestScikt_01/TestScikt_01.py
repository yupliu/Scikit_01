import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn.datasets
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor, ElasticNet

#h_data = pd.read_csv('C:\\Users\\u551896\\Source\\Repos\\ageron\\handson-ml\\datasets\\housing\\housing.csv')
#plt.plot(h_data['longitude'], h_data['latitude'])
#plt.show()

h_data = load_boston()
h_LnModel = LinearRegression()
h_SVRModel = SVR()
h_nnModel = MLPRegressor()
h_adaModel = AdaBoostRegressor()
h_GbrModel = GradientBoostingRegressor()
h_rdModel = RandomForestRegressor()
h_sgdModel = SGDRegressor()
h_elnModel = ElasticNet()
x_train, x_test, y_train, y_test = model_selection.train_test_split(h_data.data,h_data.target, test_size = 0.3)

#h_normalizer = Normalizer()
h_scaler = MinMaxScaler()
#h_data.data = h_normalizer.fit_transform(h_data.data)
h_scaler.fit(x_train)
h_scaler.transform(x_test)

h_LnModel.fit(x_train,y_train)
h_SVRModel.fit(x_train,y_train)
h_nnModel.fit(x_train,y_train)
h_adaModel.fit(x_train,y_train)
h_GbrModel.fit(x_train,y_train)
h_rdModel.fit(x_train,y_train)
h_sgdModel.fit(x_train,y_train)
h_elnModel.fit(x_train,y_train)


print(metrics.r2_score(h_LnModel.predict(x_test),y_test))
print(metrics.r2_score(h_SVRModel.predict(x_test),y_test))
print(metrics.r2_score(h_nnModel.predict(x_test),y_test))
print(metrics.r2_score(h_adaModel.predict(x_test),y_test))
print(metrics.r2_score(h_GbrModel.predict(x_test),y_test))
print(metrics.r2_score(h_rdModel.predict(x_test),y_test))
print(metrics.r2_score(h_sgdModel.predict(x_test),y_test))
print(metrics.r2_score(h_elnModel.predict(x_test),y_test))

#plt.plot(h_data.data)
#plt.show()