**CrossValidation - 223:** 

Here we went back to NewYork City rent data. Wrangling and feature engineering included, observations (rows) statistical trimming. This was done by removing extreme percentiles of ‘longitude’, ‘latitude’, and ‘Price’. We converted the ‘created’ column into to_datetime() format and split the data into train and test based on a cutoff data, with historical data for training. We cleaned up the ‘description’ column by .strip() the whitespace and replace missing values with .fillna(“”). Next created new features, ‘'has_description' by df['description'] != '', and ‘description_length’’ by df['description'].str.len(). We also used a list of all amenity columns and created a new feature named ‘perk_count’ by df[perk_cols].sum(axis=1). Additionally we created ‘days’ feature showing days on market by (df['created'] - pd.to_datetime('2016-01-01')).dt.days. Some extra features such as 'cats_and_dogs' and 'cats_or_dogs', df['rooms'] = df['bedrooms'] + df['bathrooms'] were also created. High cardinality features such as 'display_address', 'street_address', 'description' are also dropped. 

Since the number of observations are limited, we decided to use other resampling methods other than train-validation split, to utilize our limited number of observations more efficiently. For that, we used k-fold cross validation to get an estimate of a particular estimator construct. K-fold partitions the train set into k folds and uses k-1 fold for training and one fold for validation. Unlike random forest this is by default with no replacement. Moving the validation fold to all other folds will create k split sets of train-validation for us to train and evaluate on. Each surrogate model trained on each split produces a validation score. The average of all those scores represent the estimated score of future test data for that estimator construct. Once the estimator construct such as hyper parameters, pipeline steps, is decided, we refit the estimator with entire train data. That would be our final model for future predictions. The temporary surrogate models can be discarded at this point.

A pipeline estimator of OneHotEncoder(), SimpleImputer(), StandardScaler(), SelectKBest(), Ridge(), for 20 selected features, gave an average negative mean absolute error of 680  in a 3 fold cross validation using cross_val_score(), and standard deviation of 8 between the 3 splits. Using TargetEncoder(), SimpleImputer(), RandomForestRegressor() average cross_val_score() of negative MAE drops to -395. validation_curve() is used to sweep a single hyperparameter and plot the validation vs training score for tuning. RandomizedSearchCV() or GridSearchCV() were used to sweep multiple hyperparameters at the same time for tuning purposes. refit=True parameter would refit the model with all the training data for best_estimator_ parameters to get the final model. 

For the ‘Submission’ file, containing the target label prediction, we can have multiple predictions from different hyperparameter tuning and take the majority vote as the final prediction with .mode(). This is similar to the ensemble concept.

*Libraries:*
```
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from category_encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import set_config
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error
```

