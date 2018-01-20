import numpy as np
from sklearn import neighbors
import pickle
import os

# Features: type, color, season/weather, day
# Type: Shirt, Pants, Hat, Socks, Shoes, Boots, Shorts
# Color: red, Green, blue, yellow, orange, purple, indigo
# Season: Winter, Spring, Summer, Fall
# Day: Sunday, Monday, Tues, Wed, Thu, Fri, Sat

id = []
tags = []

y = []


print(len(tags))

# Assign ranks to various clothing articles
y[0] = 99
y[1] = 88
y[2] = 93
y[4] = 34
y[5] = 10
y[6] = 51
y[7] = 76
y[8] = 65
y[9] = 43

model = neighbors.KNeighborsRegressor()
model.fit(np.array(tags), np.array(y))

pickle.dump(model, os.getcwd()+'recommender')

