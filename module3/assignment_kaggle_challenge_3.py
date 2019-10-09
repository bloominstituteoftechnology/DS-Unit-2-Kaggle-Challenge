#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Kaggle Challenge, Module 3
# 
# 
# ## Assignment
# - [ ] [Review requirements for your portfolio project](https://lambdaschool.github.io/ds/unit2), then submit your dataset.
# - [ ] Continue to participate in our Kaggle challenge. 
# - [ ] Use scikit-learn for hyperparameter optimization with RandomizedSearchCV.
# - [ ] Submit your predictions to our Kaggle competition. (Go to our Kaggle InClass competition webpage. Use the blue **Submit Predictions** button to upload your CSV file. Or you can use the Kaggle API to submit your predictions.)
# - [ ] Commit your notebook to your fork of the GitHub repo.
# 
# ## Stretch Goals
# 
# ### Reading
# - Jake VanderPlas, [Python Data Science Handbook, Chapter 5.3](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html), Hyperparameters and Model Validation
# - Jake VanderPlas, [Statistics for Hackers](https://speakerdeck.com/jakevdp/statistics-for-hackers?slide=107)
# - Ron Zacharski, [A Programmer's Guide to Data Mining, Chapter 5](http://guidetodatamining.com/chapter5/), 10-fold cross validation
# - Sebastian Raschka, [A Basic Pipeline and Grid Search Setup](https://github.com/rasbt/python-machine-learning-book/blob/master/code/bonus/svm_iris_pipeline_and_gridsearch.ipynb)
# - Peter Worcester, [A Comparison of Grid Search and Randomized Search Using Scikit Learn](https://blog.usejournal.com/a-comparison-of-grid-search-and-randomized-search-using-scikit-learn-29823179bc85)
# 
# ### Doing
# - Add your own stretch goals!
# - Try other [categorical encodings](https://contrib.scikit-learn.org/categorical-encoding/). See the previous assignment notebook for details.
# - In additon to `RandomizedSearchCV`, scikit-learn has [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). Another library called scikit-optimize has [`BayesSearchCV`](https://scikit-optimize.github.io/notebooks/sklearn-gridsearchcv-replacement.html). Experiment with these alternatives.
# - _[Introduction to Machine Learning with Python](http://shop.oreilly.com/product/0636920030515.do)_ discusses options for "Grid-Searching Which Model To Use" in Chapter 6:
# 
# > You can even go further in combining GridSearchCV and Pipeline: it is also possible to search over the actual steps being performed in the pipeline (say whether to use StandardScaler or MinMaxScaler). This leads to an even bigger search space and should be considered carefully. Trying all possible solutions is usually not a viable machine learning strategy. However, here is an example comparing a RandomForestClassifier and an SVC ...
# 
# The example is shown in [the accompanying notebook](https://github.com/amueller/introduction_to_ml_with_python/blob/master/06-algorithm-chains-and-pipelines.ipynb), code cells 35-37. Could you apply this concept to your own pipelines?
# 
#%% [markdown]
# ### BONUS: Stacking!
# 
# Here's some code you can use to "stack" multiple submissions, which is another form of ensembling:
# 
# ```python
# import pandas as pd
# 
# # Filenames of your submissions you want to ensemble
# files = ['submission-01.csv', 'submission-02.csv', 'submission-03.csv']
# 
# target = 'status_group'
# submissions = (pd.read_csv(file)[[target]] for file in files)
# ensemble = pd.concat(submissions, axis='columns')
# majority_vote = ensemble.mode(axis='columns')[0]
# 
# sample_submission = pd.read_csv('sample_submission.csv')
# submission = sample_submission.copy()
# submission[target] = majority_vote
# submission.to_csv('my-ultimate-ensemble-submission.csv', index=False)
# ```

#%%
import pandas
from sklearn.model_selection import train_test_split

DATA_PATH = './data/'


train = pandas.merge(pandas.read_csv(DATA_PATH+'waterpumps/train_features.csv'), 
					pandas.read_csv(DATA_PATH+'waterpumps/train_labels.csv'))
# train_features = pandas.read_csv(DATA_PATH+'waterpumps/train_features.csv').sort_values(by='id')
# train_target = pandas.read_csv(DATA_PATH+'waterpumps/train_labels.csv').sort_values(by='id')
test = pandas.read_csv(DATA_PATH+'waterpumps/test_features.csv')
sample_submission = pandas.read_csv(DATA_PATH+'waterpumps/sample_submission.csv')

# train_features.shape, test_features.shape

#%%
def cluster(df, n_clusters=100, kmeans=None):
	from sklearn.cluster import KMeans

	if kmeans is None:
		kmeans=KMeans(n_clusters=n_clusters)
		kmeans.fit(df[['latitude', 'longitude']])
		df['cluster'] = kmeans.labels_
	else:
		df['cluster'] = kmeans.predict(df[['latitude', 'longitude']])
	return(df, kmeans)

#%%
def clean(df, n_clusters=100, kmeans=None):
	cleaned = df.copy()
	cleaned, kmeans = cluster(cleaned, n_clusters=n_clusters, kmeans=kmeans)

	cleaned['date_recorded_dt'] = pandas.to_datetime(df['date_recorded'])
	cleaned['date_recorded_ts'] = cleaned['date_recorded_dt'].view('int64')
	cleaned['month_recorded'] = cleaned['date_recorded_dt'].dt.month
	cleaned['day_recorded'] = cleaned['date_recorded_dt'].dt.day
	cleaned['year_recorded'] = cleaned['date_recorded_dt'].dt.year
	cleaned['years_in_operation'] = cleaned['year_recorded'] - cleaned['construction_year']

	return(cleaned.drop(columns=['date_recorded_dt']), kmeans)

#%%

cleaned, kmeans = clean(train)
train_features = cleaned.drop(columns=['status_group'])
train_target = cleaned['status_group']
test_features, kmeans = clean(test, kmeans=kmeans)

#%%

import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform

_oe = ce.OrdinalEncoder()
_rfc = RandomForestClassifier(random_state=3)

params = {
	'RandomForestClassifier__n_estimators': randint(3,300),
	'RandomForestClassifier__min_samples_leaf': randint(1,200),
	'RandomForestClassifier__oob_score': [True, False],
	'RandomForestClassifier__criterion': ['gini', 'entropy'],
	'RandomForestClassifier__verbose': [3]
}

# n_estimators = 1000, min_samples_leaf = 2

pipeline = Pipeline([	('OrdinalEncoder', _oe),
						('RandomForestClassifier', _rfc)])

searchCV = RandomizedSearchCV(	pipeline,
								param_distributions=params,
								n_iter=15,
								cv=3,
								scoring='accuracy',
								verbose=10,
								return_train_score=True,
								n_jobs=-1)

target_encoder = ce.OrdinalEncoder()
train_target_encoded = target_encoder.fit_transform(train_target)
train_target_encoded

searchCV.fit(train_features, train_target_encoded)

#%%
# y_val

#%%
# print(f'Validation accuracy: {pipeline.score(X_val, y_val)}')
# _rfc.score(X_val_encoded, y_val)