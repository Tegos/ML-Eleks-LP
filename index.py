import numpy as np
import urllib

import pandas
import scipy
import seaborn as seaborn
from pandas import read_csv
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns
import numpy.random as nr
from sklearn.neighbors.kde import KernelDensity
from sklearn import linear_model
from sklearn.svm import SVR

arr_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# url with dataset
url = 'zoo.csv'
# download the file
raw_data = urllib.urlopen('datasets/' + url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=',',
                     skiprows=1,
                     # converters={0: lambda x: datetime.strptime(x, "%d-%m-%Y %H-%M-%S")}
                     usecols=arr_cols
                     )

# separate the data from the target attributes
X = dataset[:, 0:15]
y = dataset[:, 16]
#
# # normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)

model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

# print (standardized_X)

# ------------ Graphs --------

# density plots
legs = X[:, 12]  # legs
legs = legs.reshape(-1, 1)

kde = KernelDensity(bandwidth=2).fit(legs)
x = np.linspace(legs.min(), legs.max(), 8).reshape(-1, 1)
density = np.exp(kde.score_samples(x))

plt.plot(x, density)
plt.plot(legs, legs * 0, 'ok', alpha=.05)
plt.ylim(-.005, .15)

plt.xlabel('Legs')
plt.ylabel('Density')
# plt.show()

# -------------------------
zoo_csv_data = read_csv('datasets/zoo.csv')
predator_g = zoo_csv_data.groupby(['predator']).size()
names = zoo_csv_data.animal_name

print predator_g
plt.figure()
plt.subplot(aspect=True)
plt.pie(predator_g, labels=['Not Predator', 'Predator'],
        colors=['green', 'red'],
        autopct='%i%%')
plt.title('Predators in the Zoo')
# plt.show()

# ---------------
# mean, median, mode,variation, range

column_names = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator',
                'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic']
value_names = ['Mean', 'Median', 'Mode', 'Range', 'Variance', 'Skewness', 'Kurtosis']

data_table = []
for x in column_names:
    temp = []
    data_column = zoo_csv_data[x]

    temp.append(np.mean(data_column))
    temp.append(np.median(data_column))
    temp.append(scipy.stats.mstats.mode(data_column))
    temp.append(np.ptp(data_column))
    temp.append(np.var(data_column))
    temp.append(scipy.stats.skew(data_column))
    temp.append(scipy.stats.kurtosis(data_column))
    data_table.append(temp)

# print pandas.DataFrame(data_table, column_names, value_names)

# ----------  Box Plots
box_plot = pandas.DataFrame(zip(X[:, 12], X[:, 13]), columns=['Count of legs', 'Has tail'])
box_plot.boxplot(column='Count of legs', by='Has tail')
# plt.show()

# ------- Pearson
for x in column_names:
    # print x
    data_column = zoo_csv_data[x]
    p_c = np.corrcoef(data_column, y)
    # print p_c

# ----- spearmanr
for x in column_names:
    # print x
    data_column = zoo_csv_data[x]
    s_c = scipy.stats.spearmanr(data_column, y)
    # print s_c

# ------------ scatter_plot
plt.figure()
seaborn.regplot(normalized_X[:, 4], y)
# plt.show()

plt.figure()
seaborn.regplot(normalized_X[:, 13], y)
# plt.show()

# --------- regression ------
x_parameter = []
y_parameter = []
for legs_data, type_class_data in zip(zoo_csv_data['legs'], zoo_csv_data['class_type']):
    x_parameter.append([float(legs_data)])
    y_parameter.append(float(type_class_data))

predict_value = 4
regr = linear_model.LinearRegression()
regr.fit(x_parameter, y_parameter)
predict_outcome = regr.predict(predict_value)
predictions = {'intercept': regr.intercept_, 'coefficient': regr.coef_, 'predicted_value': predict_outcome}

print "\nIntercept value ", predictions['intercept']
print "coefficient", predictions['coefficient']
print "Predicted value: ", predictions['predicted_value']

print ('End')
