import urllib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scipy
import sklearn
from pandas import read_csv
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors.kde import KernelDensity

arr_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

# url with dataset
url = 'datasets/zoo.csv'

# csv
zoo_csv_data = read_csv(url)

raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
data_set = np.loadtxt(
    raw_data, delimiter=',',
    skiprows=1,
    usecols=arr_cols
)

X = data_set[:, 0:15]
y = data_set[:, 15]

# normalize the data attributes
normalized_X = preprocessing.normalize(X)
normalized_y = preprocessing.normalize(y)

print (normalized_X)

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

predator_g = zoo_csv_data.groupby(['predator']).size()
names = zoo_csv_data.animal_name

# print predator_g
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
# plt.figure()
# df_box_plot = pd.DataFrame(zoo_csv_data, columns=column_names)
df_box_plot = pd.DataFrame(normalized_X, columns=column_names)
df_box_plot.plot.box().set_ylim(-0.01, 1.01)
plt.title('Box plot for digit features')
# plt.show()

# bar fins
plt.figure()
zoo_csv_data['fins'].value_counts().plot(kind='bar')
plt.title('Bar of "Fins"')
# plt.show()

# bar legs
plt.figure()
zoo_csv_data['legs'].value_counts().plot.hist(orientation='horizontal', cumulative=True)
plt.title('Histgram  of "Legs"')
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

plt.scatter(normalized_X[:, 11], normalized_y, color='DarkGreen')
plt.scatter(normalized_X[:, 14], normalized_y, color='DarkBlue')
plt.show()

print '\n--------- feature_importances --------'

model_ETS = ExtraTreesClassifier()
model_ETS.fit(X, y)

feature_importances = model_ETS.feature_importances_
index_features = np.argsort(feature_importances).tolist()
feature_importances = sorted(model_ETS.feature_importances_, reverse=True)

most_important_0 = column_names[index_features.index(0)]
most_important_1 = column_names[index_features.index(1)]

print most_important_0, ' - ', feature_importances[0]
print most_important_1, ' - ', feature_importances[1]

# Linear regression
print '--------- regression --------'
x_parameter = []
y_parameter = []

# good venomous, backbone

# Use only one feature
for legs_data, type_class_data in zip(zoo_csv_data['backbone'], zoo_csv_data['class_type']):
    x_parameter.append([float(legs_data)])
    y_parameter.append(float(type_class_data))

backbone_X = x_parameter
target = y_parameter

backbone_X_train, backbone_X_test, backbone_y_train, backbone_y_test = sklearn.cross_validation.train_test_split(
    backbone_X,
    target,
    test_size=0.05,  # for backbone,
    random_state=135
    # test_size=0.1, for venomous
    # random_state=1000
)

regr = linear_model.LinearRegression()

# Train the model
regr.fit(backbone_X_train, backbone_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(backbone_X_test) - backbone_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(backbone_X_test, backbone_y_test))

# Plot outputs
plt.figure()
plt.scatter(backbone_X_test, backbone_y_test, color='black')
plt.plot(
    backbone_X_test,
    regr.predict(backbone_X_test),
    color='blue',
    linewidth=2)
plt.xticks(())
plt.yticks(())

plt.show()
