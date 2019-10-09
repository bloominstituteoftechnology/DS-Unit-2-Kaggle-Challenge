#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Kaggle Challenge, Module 2
# 
# ## Assignment
# - [ ] Read [“Adopting a Hypothesis-Driven Workflow”](https://outline.com/5S5tsB), a blog post by a Lambda DS student about the Tanzania Waterpumps challenge.
# - [ ] Continue to participate in our Kaggle challenge.
# - [ ] Try Ordinal Encoding.
# - [ ] Try a Random Forest Classifier.
# - [ ] Submit your predictions to our Kaggle competition. (Go to our Kaggle InClass competition webpage. Use the blue **Submit Predictions** button to upload your CSV file. Or you can use the Kaggle API to submit your predictions.)
# - [ ] Commit your notebook to your fork of the GitHub repo.
# 
# ## Stretch Goals
# 
# ### Doing
# - [ ] Add your own stretch goal(s) !
# - [ ] Do more exploratory data analysis, data cleaning, feature engineering, and feature selection.
# - [ ] Try other [categorical encodings](https://contrib.scikit-learn.org/categorical-encoding/).
# - [ ] Get and plot your feature importances.
# - [ ] Make visualizations and share on Slack.
# 
# ### Reading
# 
# Top recommendations in _**bold italic:**_
# 
# #### Decision Trees
# - A Visual Introduction to Machine Learning, [Part 1: A Decision Tree](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/),  and _**[Part 2: Bias and Variance](http://www.r2d3.us/visual-intro-to-machine-learning-part-2/)**_
# - [Decision Trees: Advantages & Disadvantages](https://christophm.github.io/interpretable-ml-book/tree.html#advantages-2)
# - [How a Russian mathematician constructed a decision tree — by hand — to solve a medical problem](http://fastml.com/how-a-russian-mathematician-constructed-a-decision-tree-by-hand-to-solve-a-medical-problem/)
# - [How decision trees work](https://brohrer.github.io/how_decision_trees_work.html)
# - [Let’s Write a Decision Tree Classifier from Scratch](https://www.youtube.com/watch?v=LDRbO9a6XPU)
# 
# #### Random Forests
# - [_An Introduction to Statistical Learning_](http://www-bcf.usc.edu/~gareth/ISL/), Chapter 8: Tree-Based Methods
# - [Coloring with Random Forests](http://structuringtheunstructured.blogspot.com/2017/11/coloring-with-random-forests.html)
# - _**[Random Forests for Complete Beginners: The definitive guide to Random Forests and Decision Trees](https://victorzhou.com/blog/intro-to-random-forests/)**_
# 
# #### Categorical encoding for trees
# - [Are categorical variables getting lost in your random forests?](https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/)
# - [Beyond One-Hot: An Exploration of Categorical Variables](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)
# - _**[Categorical Features and Encoding in Decision Trees](https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931)**_
# - _**[Coursera — How to Win a Data Science Competition: Learn from Top Kagglers — Concept of mean encoding](https://www.coursera.org/lecture/competitive-data-science/concept-of-mean-encoding-b5Gxv)**_
# - [Mean (likelihood) encodings: a comprehensive study](https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study)
# - [The Mechanics of Machine Learning, Chapter 6: Categorically Speaking](https://mlbook.explained.ai/catvars.html)
# 
# #### Imposter Syndrome
# - [Effort Shock and Reward Shock (How The Karate Kid Ruined The Modern World)](http://www.tempobook.com/2014/07/09/effort-shock-and-reward-shock/)
# - [How to manage impostor syndrome in data science](https://towardsdatascience.com/how-to-manage-impostor-syndrome-in-data-science-ad814809f068)
# - ["I am not a real data scientist"](https://brohrer.github.io/imposter_syndrome.html)
# - _**[Imposter Syndrome in Data Science](https://caitlinhudon.com/2018/01/19/imposter-syndrome-in-data-science/)**_
# 
# 
# ### More Categorical Encodings
# 
# **1.** The article **[Categorical Features and Encoding in Decision Trees](https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931)** mentions 4 encodings:
# 
# - **"Categorical Encoding":** This means using the raw categorical values as-is, not encoded. Scikit-learn doesn't support this, but some tree algorithm implementations do. For example, [Catboost](https://catboost.ai/), or R's [rpart](https://cran.r-project.org/web/packages/rpart/index.html) package.
# - **Numeric Encoding:** Synonymous with Label Encoding, or "Ordinal" Encoding with random order. We can use [category_encoders.OrdinalEncoder](https://contrib.scikit-learn.org/categorical-encoding/ordinal.html).
# - **One-Hot Encoding:** We can use [category_encoders.OneHotEncoder](http://contrib.scikit-learn.org/categorical-encoding/onehot.html).
# - **Binary Encoding:** We can use [category_encoders.BinaryEncoder](http://contrib.scikit-learn.org/categorical-encoding/binary.html).
# 
# 
# **2.** The short video 
# **[Coursera — How to Win a Data Science Competition: Learn from Top Kagglers — Concept of mean encoding](https://www.coursera.org/lecture/competitive-data-science/concept-of-mean-encoding-b5Gxv)** introduces an interesting idea: use both X _and_ y to encode categoricals.
# 
# Category Encoders has multiple implementations of this general concept:
# 
# - [CatBoost Encoder](http://contrib.scikit-learn.org/categorical-encoding/catboost.html)
# - [James-Stein Encoder](http://contrib.scikit-learn.org/categorical-encoding/jamesstein.html)
# - [Leave One Out](http://contrib.scikit-learn.org/categorical-encoding/leaveoneout.html)
# - [M-estimate](http://contrib.scikit-learn.org/categorical-encoding/mestimate.html)
# - [Target Encoder](http://contrib.scikit-learn.org/categorical-encoding/targetencoder.html)
# - [Weight of Evidence](http://contrib.scikit-learn.org/categorical-encoding/woe.html)
# 
# Category Encoder's mean encoding implementations work for regression problems or binary classification problems. 
# 
# For multi-class classification problems, you will need to temporarily reformulate it as binary classification. For example:
# 
# ```python
# encoder = ce.TargetEncoder(min_samples_leaf=..., smoothing=...) # Both parameters > 1 to avoid overfitting
# X_train_encoded = encoder.fit_transform(X_train, y_train=='functional')
# X_val_encoded = encoder.transform(X_train, y_val=='functional')
# ```
# 
# For this reason, mean encoding won't work well within pipelines for multi-class classification problems.
# 
# **3.** The **[dirty_cat](https://dirty-cat.github.io/stable/)** library has a Target Encoder implementation that works with multi-class classification.
# 
# ```python
#  dirty_cat.TargetEncoder(clf_type='multiclass-clf')
# ```
# It also implements an interesting idea called ["Similarity Encoder" for dirty categories](https://www.slideshare.net/GaelVaroquaux/machine-learning-on-non-curated-data-154905090).
# 
# However, it seems like dirty_cat doesn't handle missing values or unknown categories as well as category_encoders does. And you may need to use it with one column at a time, instead of with your whole dataframe.
# 
# **4. [Embeddings](https://www.kaggle.com/learn/embeddings)** can work well with sparse / high cardinality categoricals.
# 
# _**I hope it’s not too frustrating or confusing that there’s not one “canonical” way to encode categorcals. It’s an active area of research and experimentation! Maybe you can make your own contributions!**_
#%% [markdown]
# ### Setup
# 
# You can work locally (follow the [local setup instructions](https://lambdaschool.github.io/ds/unit2/local/)) or on Colab (run the code cell below).


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
from sklearn.model_selection import KFold

_oe = ce.OrdinalEncoder()
_se = SimpleImputer()
_rfc = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, random_state=3, n_jobs=-1, verbose=3, oob_score=True, criterion='entropy')

# n_estimators = 1000, min_samples_leaf = 2

pipeline = Pipeline([	('OrdinalEncoder', _oe),
						('RandomForestClassifier', _rfc)])

#%%
train_features

#%%

from sklearn.model_selection import train_test_split

target_encoder = ce.OrdinalEncoder()
train_target_encoded = target_encoder.fit_transform(train_target)
train_target_encoded

#%%

X_train, X_val, y_train, y_val = train_test_split(train_features, train_target_encoded, random_state=2, test_size=0.1)


pipeline.fit(X_train, y_train)

#%%
# y_val

#%%
print(f'Validation accuracy: {pipeline.score(X_val, y_val)}')
# _rfc.score(X_val_encoded, y_val)


#%%
# test_features

#%%

out = test_features[['id']].copy()

#%%
out['status_group'] = pipeline.predict(test_features)

#%%
out['status_group'] = target_encoder.inverse_transform(out['status_group'])

out.sort_values(by='id').to_csv('./module2/results.csv', index=False)

#%%
