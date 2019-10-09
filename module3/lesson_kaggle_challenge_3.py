#%% [markdown]
# Lambda School Data Science
# 
# *Unit 2, Sprint 2, Module 3*
# 
# ---
#%% [markdown]
# # Kaggle Challenge, Module 3
# 
# - Do **cross-validation** with independent test set
# - Use scikit-learn for **hyperparameter optimization**
#%% [markdown]
# ### Setup
# 
# Run the code cell below. You can work locally (follow the [local setup instructions](https://lambdaschool.github.io/ds/unit2/local/)) or on Colab.
# 
# Libraries
# 
# - **category_encoders**
# - matplotlib
# - numpy
# - pandas
# - **pandas-profiling**
# - scikit-learn
# - scipy.stats

#%%
get_ipython().run_cell_magic('capture', '', "import sys\n\n# If you're on Colab:\nif 'google.colab' in sys.modules:\n    DATA_PATH = 'https://raw.githubusercontent.com/LambdaSchool/DS-Unit-2-Kaggle-Challenge/master/data/'\n    !pip install category_encoders==2.*\n    !pip install pandas-profiling==2.*\n\n# If you're working locally:\nelse:\n    DATA_PATH = '../data/'")

#%% [markdown]
# # Do cross-validation with independent test set
#%% [markdown]
# ## Overview
#%% [markdown]
# ### Predict rent in NYC 🏠
# 
# We're going back to one of our New York City real estate datasets. 

#%%
import numpy as np
import pandas as pd

# Read New York City apartment rental listing data
df = pd.read_csv(DATA_PATH+'apartments/renthop-nyc.csv')
assert df.shape == (49352, 34)

# Remove the most extreme 1% prices,
# the most extreme .1% latitudes, &
# the most extreme .1% longitudes
df = df[(df['price'] >= np.percentile(df['price'], 0.5)) & 
        (df['price'] <= np.percentile(df['price'], 99.5)) & 
        (df['latitude'] >= np.percentile(df['latitude'], 0.05)) & 
        (df['latitude'] < np.percentile(df['latitude'], 99.95)) &
        (df['longitude'] >= np.percentile(df['longitude'], 0.05)) & 
        (df['longitude'] <= np.percentile(df['longitude'], 99.95))]

# Do train/test split
# Use data from April & May 2016 to train
# Use data from June 2016 to test
df['created'] = pd.to_datetime(df['created'], infer_datetime_format=True)
cutoff = pd.to_datetime('2016-06-01')
train = df[df.created < cutoff]
test  = df[df.created >= cutoff]

# Wrangle train & test sets in the same way
def engineer_features(df):
    
    # Avoid SettingWithCopyWarning
    df = df.copy()
        
    # Does the apartment have a description?
    df['description'] = df['description'].str.strip().fillna('')
    df['has_description'] = df['description'] != ''

    # How long is the description?
    df['description_length'] = df['description'].str.len()

    # How many total perks does each apartment have?
    perk_cols = ['elevator', 'cats_allowed', 'hardwood_floors', 'dogs_allowed',
                 'doorman', 'dishwasher', 'no_fee', 'laundry_in_building',
                 'fitness_center', 'pre-war', 'laundry_in_unit', 'roof_deck',
                 'outdoor_space', 'dining_room', 'high_speed_internet', 'balcony',
                 'swimming_pool', 'new_construction', 'exclusive', 'terrace', 
                 'loft', 'garden_patio', 'common_outdoor_space', 
                 'wheelchair_access']
    df['perk_count'] = df[perk_cols].sum(axis=1)

    # Are cats or dogs allowed?
    df['cats_or_dogs'] = (df['cats_allowed']==1) | (df['dogs_allowed']==1)

    # Are cats and dogs allowed?
    df['cats_and_dogs'] = (df['cats_allowed']==1) & (df['dogs_allowed']==1)

    # Total number of rooms (beds + baths)
    df['rooms'] = df['bedrooms'] + df['bathrooms']
    
    # Extract number of days elapsed in year, and drop original date feature
    df['days'] = (df['created'] - pd.to_datetime('2016-01-01')).dt.days
    df = df.drop(columns='created')

    return df

train = engineer_features(train)
test = engineer_features(test)


#%%
import pandas_profiling
train.profile_report()

#%% [markdown]
# ### Validation options
#%% [markdown]
# Let's take another look at [Sebastian Raschka's diagram of model evaluation methods.](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html) So far we've been using "**train/validation/test split**", but we have more options. 
# 
# Today we'll learn about "k-fold **cross-validation** with independent test set", for "model selection (**hyperparameter optimization**) and performance estimation."
# 
# <img src="https://sebastianraschka.com/images/blog/2018/model-evaluation-selection-part4/model-eval-conclusions.jpg" width="600">
# 
# <sup>Source: https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html</sup>
# 
# 
#%% [markdown]
# ### Cross-validation: What & Why?
#%% [markdown]
# The Scikit-Learn docs show a diagram of how k-fold cross-validation works, and explain the pros & cons of cross-validation versus train/validate/test split.
# 
# #### [Scikit-Learn User Guide, 3.1 Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
# 
# > When evaluating different settings (“hyperparameters”) for estimators, there is still a risk of overfitting on the test set because the parameters can be tweaked until the estimator performs optimally. This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be held out as a so-called “validation set”: training proceeds on the training set, after which evaluation is done on the validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.
# >
# > However, **by partitioning the available data into three sets, we drastically reduce the number of samples which can be used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.**
# >
# > **A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV.** 
# 
# <img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" width="600">
# 
# > In the basic approach, called k-fold CV, the training set is split into k smaller sets. The following procedure is followed for each of the k “folds”:
# >
# > - A model is trained using $k-1$ of the folds as training data;
# > - the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).
# >
# > The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. **This approach can be computationally expensive, but does not waste too much data (as is the case when fixing an arbitrary validation set).**
#%% [markdown]
# ## Follow Along
#%% [markdown]
# ### cross_val_score
# 
# How do we get started? According to the [Scikit-Learn User Guide](https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics), 
# 
# > The simplest way to use cross-validation is to call the [**`cross_val_score`**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) helper function
# 
# But, there's a quirk: For scikit-learn's cross-validation [**scoring**](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter), higher is better. But for regression error metrics, lower is better. So scikit-learn multiplies regression error metrics by -1 to make them negative. That's why the value of the `scoring` parameter is `'neg_mean_absolute_error'`.
# 
# So, k-fold cross-validation with this dataset looks like this:
#%% [markdown]
# ### Linear Model

#%%
import category_encoders as ce
import numpy as np
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

target = 'price'
high_cardinality = ['display_address', 'street_address', 'description']
features = train.columns.drop([target] + high_cardinality)
X_train = train[features]
y_train = train[target]

pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True), 
    SimpleImputer(strategy='mean'), 
    StandardScaler(), 
    SelectKBest(f_regression, k=20), 
    Ridge(alpha=1.0)
)

k = 3
scores = cross_val_score(pipeline, X_train, y_train, cv=k, 
                         scoring='neg_mean_absolute_error')
print(f'MAE for {k} folds:', -scores)


#%%
-scores.mean()

#%% [markdown]
# ### Random Forest

#%%
from sklearn.ensemble import RandomForestRegressor

features = train.columns.drop(target)
X_train = train[features]
y_train = train[target]

pipeline = make_pipeline(
    ce.TargetEncoder(min_samples_leaf=1, smoothing=1), 
    SimpleImputer(strategy='median'), 
    RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
)

k = 3
scores = cross_val_score(pipeline, X_train, y_train, cv=k, 
                         scoring='neg_mean_absolute_error')
print(f'MAE for {k} folds:', -scores)


#%%
-scores.mean()

#%% [markdown]
# But the Random Forest has many hyperparameters. We mostly used the defaults, and arbitrarily chose `n_estimators`. Is it too high? Too low? Just right? How do we know?

#%%
print('Model Hyperparameters:')
print(pipeline.named_steps['randomforestregressor'])

#%% [markdown]
# ## Challenge
# 
# You will continue to participate in our Kaggle challenge. Use cross-validation and submit new predictions.
#%% [markdown]
# # Use scikit-learn for hyperparameter optimization
#%% [markdown]
# ## Overview
#%% [markdown]
# "The universal tension in machine learning is between optimization and generalization; the ideal model is one that stands right at the border between underfitting and overfitting; between undercapacity and overcapacity. To figure out where this border lies, first you must cross it." —[Francois Chollet](https://books.google.com/books?id=dadfDwAAQBAJ&pg=PA114)
#%% [markdown]
# ### Validation Curve
# 
# Let's try different parameter values, and visualize "the border between underfitting and overfitting." 
# 
# Using scikit-learn, we can make [validation curves](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html), "to determine training and test scores for varying parameter values. This is similar to grid search with one parameter."
#%% [markdown]
# <img src="https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.03-validation-curve.png">
# 
# <sup>Source: https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html#Validation-curves-in-Scikit-Learn</sup>
#%% [markdown]
# Validation curves are awesome for learning about overfitting and underfitting. (But less useful in real-world projects, because we usually want to vary more than one parameter.)
# 
# For this example, let's see what happens when we vary the depth of a decision tree. (This will be faster than varying the number of estimators in a random forest.)

#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor

pipeline = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(), 
    DecisionTreeRegressor()
)

depth = range(1, 30, 3)
train_scores, val_scores = validation_curve(
    pipeline, X_train, y_train,
    param_name='decisiontreeregressor__max_depth', 
    param_range=depth, scoring='neg_mean_absolute_error', 
    cv=3,
    n_jobs=-1
)

plt.figure(dpi=150)
plt.plot(depth, np.mean(-train_scores, axis=1), color='blue', label='training error')
plt.plot(depth, np.mean(-val_scores, axis=1), color='red', label='validation error')
plt.title('Validation Curve')
plt.xlabel('model complexity: RandomForestRegressor max_depth')
plt.ylabel('model score: Mean Absolute Error')
plt.legend();


#%%
plt.figure(dpi=150)
plt.plot(depth, np.mean(-train_scores, axis=1), color='blue', label='training error')
plt.plot(depth, np.mean(-val_scores, axis=1), color='red', label='validation error')
plt.title('Validation Curve, Zoomed In')
plt.xlabel('model complexity: RandomForestRegressor max_depth')
plt.ylabel('model score: Mean Absolute Error')
plt.ylim((500, 700))  # Zoom in
plt.legend();

#%% [markdown]
# ## Follow Along
#%% [markdown]
# To vary multiple hyperparameters and find their optimal values, let's try **Randomized Search CV.**
#%% [markdown]
# #### [Scikit-Learn User Guide, 3.2 Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
# 
# > Hyper-parameters are parameters that are not directly learnt within estimators. In scikit-learn they are passed as arguments to the constructor of the estimator classes. 
# >
# > It is possible and recommended to search the hyper-parameter space for the best cross validation score.
# >
# > [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) exhaustively considers all parameter combinations, while [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) can sample a given number of candidates from a parameter space with a specified distribution. 
# >
# > While using a grid of parameter settings is currently the most widely used method for parameter optimization, other search methods have more favourable properties. [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values.
# >
# > Specifying how parameters should be sampled is done using a dictionary. Additionally, a computation budget, being the number of sampled candidates or sampling iterations, is specified using the `n_iter` parameter. 
# >
# > For each parameter, either a distribution over possible values or a list of discrete choices (which will be sampled uniformly) can be specified.
#%% [markdown]
# Here's a good blog post to explain more: [**A Comparison of Grid Search and Randomized Search Using Scikit Learn**](https://blog.usejournal.com/a-comparison-of-grid-search-and-randomized-search-using-scikit-learn-29823179bc85).
# 
# <img src="https://miro.medium.com/max/2500/1*9W1MrRkHi0YFmBoHi9Y2Ow.png" width="50%">
#%% [markdown]
# ### Linear Model

#%%
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

features = train.columns.drop([target] + high_cardinality)
X_train = train[features]
y_train = train[target]

pipeline = make_pipeline(
    ce.OneHotEncoder(use_cat_names=True), 
    SimpleImputer(), 
    StandardScaler(), 
    SelectKBest(f_regression), 
    Ridge()
)

param_distributions = {
    'simpleimputer__strategy': ['mean', 'median'], 
    'selectkbest__k': range(1, len(X_train.columns)+1), 
    'ridge__alpha': [0.1, 1, 10], 
}

# If you're on Colab, decrease n_iter & cv parameters
search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_distributions, 
    n_iter=100, 
    cv=5, 
    scoring='neg_mean_absolute_error', 
    verbose=10, 
    return_train_score=True, 
    n_jobs=-1
)

search.fit(X_train, y_train);


#%%
print('Best hyperparameters', search.best_params_)
print('Cross-validation MAE', -search.best_score_)


#%%
# If we used GridSearchCV instead of RandomizedSearchCV, 
# how many candidates would there be?
# 2 imputation strategies * n columns * 3 Ridge alphas
2 * len(X_train.columns) * 3

#%% [markdown]
# ### "Fitting X folds for each of Y candidates, totalling Z fits" ?
# 
# What did that mean? What do you think?
# 
# 
#%% [markdown]
# ### Random Forest
#%% [markdown]
# #### [Scikit-Learn User Guide, 3.2 Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
# 
# > [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values.
# >
# > For each parameter, either a distribution over possible values or a list of discrete choices (which will be sampled uniformly) can be specified.
# >
# > This example uses the `scipy.stats` module, which contains many useful distributions for sampling parameters.

#%%
from scipy.stats import randint, uniform

features = train.columns.drop(target)
X_train = train[features]
y_train = train[target]

pipeline = make_pipeline(
    ce.TargetEncoder(), 
    SimpleImputer(), 
    RandomForestRegressor(random_state=42)
)

param_distributions = {
    'targetencoder__min_samples_leaf': randint(1, 1000), 
    'targetencoder__smoothing': uniform(1, 1000), 
    'simpleimputer__strategy': ['mean', 'median'], 
    'randomforestregressor__n_estimators': randint(50, 500), 
    'randomforestregressor__max_depth': [5, 10, 15, 20, None], 
    'randomforestregressor__max_features': uniform(0, 1), 
}

# If you're on Colab, decrease n_iter & cv parameters
search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_distributions, 
    n_iter=10, 
    cv=3, 
    scoring='neg_mean_absolute_error', 
    verbose=10, 
    return_train_score=True, 
    n_jobs=-1
)

search.fit(X_train, y_train);


#%%
print('Best hyperparameters', search.best_params_)
print('Cross-validation MAE', -search.best_score_)

#%% [markdown]
# ### See detailed results

#%%
pd.DataFrame(search.cv_results_).sort_values(by='rank_test_score')

#%% [markdown]
# ### Make predictions for test set

#%%
pipeline = search.best_estimator_


#%%
from sklearn.metrics import mean_absolute_error

X_test = test[features]
y_test = test[target]

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: ${mae:,.0f}')

#%% [markdown]
# ## Challenge
# 
# For your assignment, use scikit-learn for hyperparameter optimization with RandomizedSearchCV.
#%% [markdown]
# # Review
# 
# Continue to participate in our Kaggle Challenge, and practice these objectives:
# 
# - Do **cross-validation** with independent test set
# - Use scikit-learn for **hyperparameter optimization**
# 
# You can refer to these suggestions when you do hyperparameter optimization, now and in future projects:
#%% [markdown]
# ### Tree Ensemble hyperparameter suggestions
# 
# #### Random Forest
# - class_weight (for imbalanced classes)
# - max_depth (usually high, can try decreasing)
# - n_estimators (too low underfits, too high wastes time)
# - min_samples_leaf (increase if overfitting)
# - max_features (decrease for more diverse trees)
# 
# #### XGBoost
# - scale_pos_weight (for imbalanced classes)
# - max_depth (usually low, can try increasing)
# - n_estimators (too low underfits, too high wastes time/overfits) — _I recommend using early stopping instead of cross-validation_
# - learning_rate (too low underfits, too high overfits)
# - See [Notes on Parameter Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html) and [DART booster](https://xgboost.readthedocs.io/en/latest/tutorials/dart.html) for more ideas
# 
# ### Linear Model hyperparameter suggestions 
# 
# #### Logistic Regression
# - C
# - class_weight (for imbalanced classes)
# - penalty
# 
# #### Ridge / Lasso Regression
# - alpha
# 
# #### ElasticNet Regression
# - alpha
# - l1_ratio
# 
# For more explanation, see [**Aaron Gallant's 9 minute video on Ridge Regression**](https://www.youtube.com/watch?v=XK5jkedy17w)!
#%% [markdown]
# # Sources
# - Jake VanderPlas, [Python Data Science Handbook, Chapter 5.3,](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html) Hyperparameters and Model Validation
# - Peter Worcester, [A Comparison of Grid Search and Randomized Search Using Scikit Learn](https://blog.usejournal.com/a-comparison-of-grid-search-and-randomized-search-using-scikit-learn-29823179bc85)
# - Ron Zacharski, [A Programmer’s Guide to Data Mining, Chapter 5,](http://guidetodatamining.com/chapter5/) first 10 pages, for a great explanation of cross-validation with examples and pictures
# - Sebastian Raschka, [Model Evaluation](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html)
# - [Scikit-Learn User Guide, 3.1 Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
# - [Scikit-Learn User Guide, 3.2 Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
# - [sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
# - [sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
# - [xgboost, Notes on Parameter Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)

