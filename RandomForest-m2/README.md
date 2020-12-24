**RandomForestClassifer - 222:** 

Tanzania water pump data set was used again. Data wrangling, consisted of dropping categorical features with cardinality above 50, replacing longitude and latitude coordinates of zero and some other columns with zero values with np.nan, converting ‘date_recorded’ column to datatime dtype, engineering new features such as date_in service, splitting date to year/month/day features, creating year_missing, and other column missing value columns. Dropping unusual variance columns like id and constant value columns, dropping duplicate columns. Labeled data was split to train and validation set in addition to unlabeled test data. To convert categorical data to numerical, we used two different encoder. OneHotEncoder, SimpleImputer, SelectKBest, RandomForestClassifier. OneHotEncoder expands number of features and needed to be accompanied by SelectKBest to limit the computing resources. As a drawback, selecting a few features among all features would reduce the accuracy of the estimator to %74. The target was a three class label ‘status_group’. On  the other side, a pipeline of OrdinalEncoder, SimpleImputer, RandomForestClassifier, would maintain the same number of features and did not need a SelectKBest. This led to a higher validation accuracy of 80%. For a tree based classifier, ordinal encoder provide the same category level distinction as OneHotEncoder, and do not cost an accuracy penalty. However, for LogisticRegression, OneHotEncoder expands the number of coefficients and provides a better fit compared to Ordinal one. Next we examined the effect of max_depth and n_estimators hyperparameters on golf puttings data set. It shows that increasing max_depth moves towards overfitting and increasing n_estimators helps to regularize the model and make it less sensitive to noise data and predicting less variance output. We implemented the bagging manually. RandomForest works based on bagging and aggregation. Bagging is randomly sampling a subset of training data. If it’s with replacement, it’s called bootstrapping. One tree for each bag of data are trained . Then the predicted output of validation data for all the trained trees are aggregated (for classifier it’s majority, and for regression problem is mean). If the max_features are limited then for each tree a random set of features are also selected. To replicate exactly the behaviour of a single tree in RandomForestClassifier(), we should use  bootstrap=False and max_features=None, n_estimators=1. 

*Libraries:*
```
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
import graphviz
from ipywidgets import interact
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
```

https://github.com/skhabiri/PredictiveModeling-TreeBasedModels-u2s2/tree/master/RandomForest-m2
