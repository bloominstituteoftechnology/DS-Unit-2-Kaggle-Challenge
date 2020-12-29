**ClassificationMetrics - 224:**

Continued working with Tanzania Waterpump. 

Wrangling the data included converting “date_recorded” to .to_datetime(), and extract the year, month and day with dt.year and so on. Creating “year” of service as a new feature. Getting rid of unusable valiances (constant and changing with every sample), and duplicate columns. The zero filled values for latitude and longitude were also replaced by np.nan. 

Next a pipeline of FunctionTransformer(), OrdinalEncoder(), SimpleImputer(), and RandomForestClassifier() are made to wrangle, encode, impute and classify the train, validation and test sets. We get a validation accuracy of 0.81. 

Next we use plot_confusion_matrix() to plot the confusion matrix, which is the True Labels vs. Predicted Labels. Based on the Confusion Matrix or printing the classification report we obtain the accuracy, precision and recall metrics of the classifier for default probability threshold of 0.5. Recall is TP/(All Positives), Precision is TP/(All predicted as Positive), Accuracy is (TP+TN)/(All Predictions). 2/F1_score can be calculated as (1/Recall + 1/Precision). The recall for nonfunctional pumps is 0.79. Our test size is 6000 samples. 

Our goal is to find as many nonfunctional pumps as in 2000 visits to repair them. Hence we need to increase the classifier Precision at the expense of Recall.  The target has three labels of “functional”, “non-functional” and “needs repair”. We merge the latter two to create a binary classification, to be able to adjust the prediction probability. Based on the value_counts() of the training set, a random inspection will yield a %46 chance of visiting a Not-functional site. After fitting the previous pipeline estimator and getting the classification report we get 0.84 score for Positive label (not-functional) Precision. We plot .distplot() and .histplot() of .predict_proba of positive class, to visualize how far we can increase probability threshold level while still able to get 2000 positive label predictions. In other words, for a given threshold level, sum(pipeline.predict_proba(X_val)[:, 1] > threshold) need to be higher than 2000. Using .sort_values() on .predict_proba results we select the top 2000 predictions for positive class. It turns out to be with a confidence level of at least 0.93 probability. Consequently the precision is improved to 0.986. 

By sweeping the probability threshold we can plot the ROC-AUC curve which is TPR=TP/(all positives) vs FPR=FP/(All negatives). For a binary classifier when we move the probability threshold over the distribution plot of one class, If we can place the threshold such to separate both classes that is AUC of 1 and it's an ideal binary classifier. If both distributions overlap each other, the AUC is 0.5 and it's a random classifier with baseline of 50% accuracy. We use roc_curve() from sklearn to get the tpr and fpr values and use scatter plot to plot ROC-AUC. Using roc_auc_score, the area under the curve is 0.9.

**Libraries:**
```
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
```

https://github.com/skhabiri/PredictiveModeling-TreeBasedModels-u2s2/tree/master/ClassificationMetrics-m4
