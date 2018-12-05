import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

h_data = pd.read_csv('C:\\Users\\u551896\\Source\\Repos\\ageron\\handson-ml\\datasets\\housing\\housing.csv')
plt.plot(h_data['longitude'], h_data['latitude'])
plt.show()