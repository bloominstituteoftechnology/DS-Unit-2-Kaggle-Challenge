#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Kaggle Challenge, Module 4
# 
# ## Assignment
# - [ ] If you haven't yet, [review requirements for your portfolio project](https://lambdaschool.github.io/ds/unit2), then submit your dataset.
# - [ ] Plot a confusion matrix for your Tanzania Waterpumps model.
# - [ ] Continue to participate in our Kaggle challenge. Every student should have made at least one submission that scores at least 60% accuracy (above the majority class baseline).
# - [ ] Submit your final predictions to our Kaggle competition. Optionally, go to **My Submissions**, and _"you may select up to 1 submission to be used to count towards your final leaderboard score."_
# - [ ] Commit your notebook to your fork of the GitHub repo.
# - [ ] Read [Maximizing Scarce Maintenance Resources with Data: Applying predictive modeling, precision at k, and clustering to optimize impact](https://towardsdatascience.com/maximizing-scarce-maintenance-resources-with-data-8f3491133050), by Lambda DS3 student Michael Brady. His blog post extends the Tanzania Waterpumps scenario, far beyond what's in the lecture notebook.
# 
# 
# ## Stretch Goals
# 
# ### Reading
# - [Attacking discrimination with smarter machine learning](https://research.google.com/bigpicture/attacking-discrimination-in-ml/), by Google Research, with  interactive visualizations. _"A threshold classifier essentially makes a yes/no decision, putting things in one category or another. We look at how these classifiers work, ways they can potentially be unfair, and how you might turn an unfair classifier into a fairer one. As an illustrative example, we focus on loan granting scenarios where a bank may grant or deny a loan based on a single, automatically computed number such as a credit score."_
# - [Notebook about how to calculate expected value from a confusion matrix by treating it as a cost-benefit matrix](https://github.com/podopie/DAT18NYC/blob/master/classes/13-expected_value_cost_benefit_analysis.ipynb)
# - [Simple guide to confusion matrix terminology](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) by Kevin Markham, with video
# - [Visualizing Machine Learning Thresholds to Make Better Business Decisions](https://blog.insightdatascience.com/visualizing-machine-learning-thresholds-to-make-better-business-decisions-4ab07f823415)
# 
# 
# ### Doing
# - [ ] Share visualizations in our Slack channel!
# - [ ] RandomizedSearchCV / GridSearchCV, for model selection. (See module 3 assignment notebook)
# - [ ] More Categorical Encoding. (See module 2 assignment notebook)
# - [ ] Stacking Ensemble. (See below)
# 
# ### Stacking Ensemble
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
# submissions = (pandas.read_csv(file)[[target]] for file in files)
# ensemble = pandas.concat(submissions, axis='columns')
# majority_vote = ensemble.mode(axis='columns')[0]
# 
# sample_submission = pandas.read_csv('sample_submission.csv')
# submission = sample_submission.copy()
# submission[target] = majority_vote
# submission.to_csv('my-ultimate-ensemble-submission.csv', index=False)
# ```

#%%
import pandas

DATA_PATH = './data/'

# Merge train_features.csv & train_labels.csv
train = pandas.merge(pandas.read_csv(DATA_PATH+'waterpumps/train_features.csv'), 
                 pandas.read_csv(DATA_PATH+'waterpumps/train_labels.csv'))

# Read test_features.csv & sample_submission.csv
test = pandas.read_csv(DATA_PATH+'waterpumps/test_features.csv')
sample_submission = pandas.read_csv(DATA_PATH+'waterpumps/sample_submission.csv')


#%%
from typing import Optional

def keepTopN(	column:pandas.Series,
				n:int,
				default:Optional[object] = None) -> pandas.Series:
	"""
	Keeps the top n most popular values of a Series, while replacing the rest with `default`
	
	Args:
		column (pandas.Series): Series to operate on
		n (int): How many values to keep
		default (object, optional): Defaults to NaN. Value with which to replace remaining values
	
	Returns:
		pandas.Series: Series with the most popular n values
	"""
	import numpy

	if default is None: default = numpy.nan

	val_counts = column.value_counts()
	if n > len(val_counts): n = len(val_counts)
	top_n = list(val_counts[:n].index)
	return(column.where(column.isin(top_n), other=default))

def oneHot(	frame:pandas.DataFrame, 
			cols:Optional[list] = None,
			exclude_cols:Optional[list] = None,
			max_cardinality:Optional[int] = None,
			required_out_cols:Optional[list] = None) -> pandas.DataFrame:
	"""
	One-hot encodes the dataframe.
	
	Args:
		frame (pandas.DataFrame): Dataframe to clean
		cols (list, optional): Columns to one-hot encode. Defaults to all string columns.
		exclude_cols (list, optional): Columns to skip one-hot encoding. Defaults to None.
		max_cardinality (int, optional): Maximum cardinality of columns to encode. Defaults to no maximum cardinality.
	
	Returns:
		pandas.DataFrame: The one_hot_encoded dataframe.
	"""
	import category_encoders
	import numpy

	one_hot_encoded = frame.copy()

	if cols is None: cols = list(one_hot_encoded.columns[one_hot_encoded.dtypes=='object'])

	if exclude_cols is not None:
		for col in exclude_cols:
			cols.remove(col)

	if max_cardinality is not None:
		described = one_hot_encoded[cols].describe(exclude=[numpy.number])
		cols = list(described.columns[described.loc['unique'] <= max_cardinality])

	encoder = category_encoders.OneHotEncoder(return_df=True, use_cat_names=True, cols=cols)
	one_hot_encoded = encoder.fit_transform(one_hot_encoded)

	if required_out_cols is not None:
		for column in set(required_out_cols) - set(one_hot_encoded.columns):
			one_hot_encoded[column] = numpy.zeros(one_hot_encoded.shape[0])

	return(one_hot_encoded)

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

def clean(df, n_clusters=250, kmeans=None, n=5, required_out_cols=None, exclude_cols=None):

	cleaned = df.copy()
	cleaned, kmeans = cluster(cleaned, n_clusters=n_clusters, kmeans=kmeans)

	cleaned['date_recorded_dt'] = pandas.to_datetime(df['date_recorded'])
	cleaned['date_recorded_ts'] = cleaned['date_recorded_dt'].view('int64')
	cleaned['month_recorded'] = cleaned['date_recorded_dt'].dt.month
	cleaned['day_recorded'] = cleaned['date_recorded_dt'].dt.day
	cleaned['year_recorded'] = cleaned['date_recorded_dt'].dt.year
	cleaned['years_in_operation'] = cleaned['year_recorded'] - cleaned['construction_year']
	cleaned = cleaned.drop(columns = ['date_recorded'])

	for column in cleaned.columns[cleaned.dtypes=='object']:
		cleaned[column] = keepTopN(cleaned[column], n=n, default='other')

	encoded = oneHot(cleaned.drop(columns=['date_recorded_dt']), exclude_cols=exclude_cols, max_cardinality=n+1, required_out_cols=required_out_cols)

	return(encoded, kmeans)

#%%

cleaned, kmeans = clean(train, exclude_cols=['status_group'])
train_features = cleaned.drop(columns=['status_group'])
train_target = cleaned['status_group'].values.flatten()
test_features, kmeans = clean(test, kmeans=kmeans, required_out_cols=list(train_features.columns))

#%%
print(list(train_features.columns))

#%%
print(list(test_features.columns))

#%%
train_features.dtypes[train_features.dtypes=='object']

#%%

import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_ss = StandardScaler()
_pca = PCA()
_rfc = RandomForestClassifier(random_state=3)

params = {
	'RandomForestClassifier__n_estimators': [27,270],
	'RandomForestClassifier__min_samples_leaf': [5,10],
	'RandomForestClassifier__oob_score': [False],
	'RandomForestClassifier__criterion': ['gini'],
	'PCA__n_components': [40,80,160]
}

# n_estimators = 1000, min_samples_leaf = 2

pipeline = Pipeline([	('StandardScaler', _ss),
						('PCA', _pca),
						('RandomForestClassifier', _rfc)])

searchCV = RandomizedSearchCV(	pipeline,
								param_distributions=params,
								n_iter=5,
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
print('Cross-validation accuracy', searchCV.best_score_)
print('Best hyperparameters', searchCV.best_params_)

#%%
out = test_features[['id']].copy()

#%%
train_features.shape

#%%
test_features.shape

#%%
test_features_n = test_features.drop(columns=list(set(test_features.columns) - set(train_features.columns)))

#%%

#%%
out['status_group'] = searchCV.predict(test_features_n)

#%%
out['status_group'].value_counts()

#%%
out['status_group'] = target_encoder.inverse_transform(out['status_group'])

out.sort_values(by='id').to_csv('./module4/results.csv', index=False)

#%%

