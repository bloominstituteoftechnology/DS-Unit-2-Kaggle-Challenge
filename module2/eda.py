import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import f_regression, SelectKBest, f_classif
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def correlations(data, y, xs):
    """
    Computes Pearsons and Spearman correlation coefficient.

    Parameters
    ----------
    data: Pandas Data Frame
    y: Target/Dependent variable - has to be python string object
    xs: Features/Independent variables - python list of string objects

    Returns
    ------
    df: Pandas Data Frame Object
    """
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    if (y is None) or (xs is None):
        raise ValueError("The parameter `y` or `xs` has to be non-nil reference")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` - has to be Pandas DataFrame object")
    if not isinstrance(y, str):
        raise ValueError("`data` - has to be Python string object")
    if not isinstrance(xs, list):
        raise ValueError("`xs` - has to be Python list object")    
    
    rs = []
    rhos = []
    for x in xs:
        r = stats.pearsonr(data[y], data[x])[0]
        rs.append(r)
        rho = stats.spearmanr(data[y], data[x])[0]
        rhos.append(rho)
    return pd.DataFrame({"feature": xs, "r": rs, "rho": rhos})


def get_acc_rec_fone(y_true, y_pred, verbose=False, title='TEST'):
    """
    Returns results dictionary containing accuracy score, precision, recall, f1 score, and support. If verbose is set True, than prints out results.

    Parameters
    ----------
    y_true : Python list or numpy 1D array
    y_pred : Python list or numpy 1D array

    Returns
    -------
    results : Python dictionary
    """
    if (y_true is None) or (y_pred is None):
        raise ValueError('Parameters `y_true` and `y_pred` must be a non-nil reference to python list or numpy arrary') 
    # Assert dims match
    assert y_true.shape == y_pred.shape
    # Results dict
    results = {}
    # Compute scores
    acc_ = accuracy_score(y_true, y_pred)
    prec_, rec_, fscore_, sprt_ = score(y_true, y_pred)
    # Store scores
    results['ACCURACY'] = acc_
    results['PRECISION'] = prec_
    results['RECALL'] = rec_
    results['F1'] = fscore_
    results['SUPPORT'] = sprt_
    df = pd.DataFrame({
        'Precision':prec_,
        'Recall':rec_,
        'F Score':fscore_,
        'Support':sprt_
    }, index=y_train.unique())
    if verbose:
        print(f'-------- {title} SET --------')
        print(f'Accuracy Score: {acc_:.2f}')
        print(df)
    return results

def get_clf_metrics(model, X_train, X_val, y_train, y_val, verbose=False):
    """
    Returns a result dictionary containing both training and validation scores of classifer.
    If verbose is set True, prints out the results.
    """
    if not model:
        raise ValueError("model has to be a non-nil parameter")
    
    # Sanity check the columns match for both validation and training set
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    assert X_train.shape[1] == X_val.shape[1]
    
    # Results dict
    results = {}

    # Compute Training accuracy
    y_true = y_train
    y_pred = model.predict(X_train)
    results['Train'] = get_acc_rec_fone(
        y_true, y_pred, verbose=verbose, title='TRAINING')

    # Validation Accuracy
    y_true = y_val
    y_pred = model.predict(X_val)
    results['Validation'] = get_acc_rec_fone(
        y_true, y_pred, verbose=verbose, title='VALIDATION')
    return results


def get_rmse_mae_r2(y_true, y_pred, verbose=False, title='TEST'):
    """
    Returns regression metrics like - R^2, MSE, RMSE, and MAE as dictionary,
    and prints them out if verbose is set to True.

    Parameters
    ----------
    y_true : Python list or numpy 1D array
    y_pred : Python list or numpy 1D array

    Returns
    -------
    results : Python dictionary
    """
    if (y_true is None) or (y_pred is None):
        raise ValueError('Parameters `y_true` and `y_pred` must be a non-nil reference to python list or numpy arrary')
    
    assert y_true.shape == y_pred.shape
    # Results dict
    results = {}
    mae_ = mean_absolute_error(y_true, y_pred)
    mse_ = mean_squared_error(y_true, y_pred)
    rmse_ = np.sqrt(mse)
    r_square_ = r2_score(y_true, y_pred)
    # Store scores
    results['R_SQUARE'] = r_square_
    results['MAE'] = mae_
    results['MSE'] = mse_
    results['RMSE'] = rmse_
    if verbose:
        print(f'-------- {title} SET --------')
        print(f'R^2: {r_square:.2f}')
        print(f'MSE: {mse:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'MAE: {mae:.2f}')
    return results


def get_reg_metrics(model, X_train, X_val, y_train, y_val, verbose=False):
    """
    Returns a result dictionary containing both training and validation metrics for regression model. 
    If verbose is set True, prints out the results.
    """
    if not model:
        raise ValueError("model has to be a non-nil parameter")

    assert X_train.shape[1] == X_val.shape[1]

    # Results dict
    results = {}
    # Compute Training accuracy
    y_true = y_train
    y_pred = model.predict(X_train)
    results['Train'] = get_rmse_mae_r2(
        y_true, y_pred, verbose=verbose, title='TRAINING')

    # Validation Accuracy
    y_true = y_val
    y_pred = model.predict(X_val)
    results['Validation'] = get_rmse_mae_r2(
        y_true, y_pred, verbose=verbose, title='VALIDATION')
    return results

def find_k_best_features(model, X_train, X_val, y_train, y_val):
    """
    Returns a pandas dataframe with incrementing K features and their metrics for either regeression task or classification task.

    Parameters
    ----------
    model : sklearn model
    X_train : Pandas Dataframe of training set features
    X_val : Pandas Dataframe of validation set features
    y_train : Pandas Series of target column from training set
    y_val : Pandas Series of target column from validation set
    model_type : Default is 'reg' for Regression, change to 'clf' for Classification task

    Returns
    -------
    results : Pandas DataFrame containing K best features and their metrics
    """
    if not model:
        raise ValueError("model has to be a non-nil parameter")
    
    # Make sure columns match for both training set and validation set
    assert X_train.shape[1] == X_val.shape[1]
    
    # Get model type
    model_type = getattr(model, "_estimator_type", None) 
    
    # Store results
    results = {}
    results['K'] = []
 
    if model_type == 'regressor':
        results['R_SQUARE'] = []
        results['MSE'] = []
        results['RMSE'] = []
        results['MAE'] = []
    elif model_type == 'classifier':
        results['ACCURACY'] = []
        results['F1'] = []
    else:
        raise ValueError("Incorrect option was selected for `model_type`, can only be 'reg' or 'clf'")

    # Loop through all the columns
    for k in range(1, len(X_train.columns)+1):
        # Store k num
        results['K'].append(k)

        # Regression task
        if model_type == 'regressor':
            # Select k feature from training set
            selector = SelectKBest(score_func=f_regression, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val)
            # Get predicted values
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_val_selected)
            # Compute metrics
            metrics = get_rmse_mae_r2(y_val, y_pred)
            # Store metrics
            results['R_SQUARE'].append(metrics['R_SQUARE'])
            results['MSE'].append(metrics['MSE'])
            results['RMSE'].append(metrics['RMSE'])
            results['MAE'].append(metrics['MAE'])
    
        # Classification task
        elif model_type == 'classifier':
            # Select k feature from training set
            selector = SelectKBest(score_func=f_classif, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val)
            # Get predicted values
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_val_selected)
            # Compute metrics
            metrics = get_acc_rec_fone(y_val, y_pred)
            # Store metrics
            results['ACCURACY'].append(metrics['ACCURACY'])
            results['F1'].append(metrics['F1'])
        else:
            raise ValueError('This logic shouldn\'t have been executed')

    if model_type == 'regressor':
        return pd.DataFrame(data={
        'K':results['K'],
        'R_SQUARE':results['R_SQUARE'],
        'MAE':results['MAE'],
        'MSE':results['MSE'],
        'RMSE':results['RMSE']
    })
    elif model_type == 'classifier':
        return pd.DataFrame(data={
        'K':results['K'],
        'ACCURACY':results['ACCURACY'],
        'F1':results['F1']
    })
    else:
        return None

def get_categorical_columns(df):
    """
    Returns categorical columns from pandas dataframe

    Parameters
    ----------
    df : Pandas Dataframe

    Returns
    -------
    Python list
    """
    if df is None:
        raise ValueError(
            "The parameter 'df' must be assigned a non-nil reference to a Pandas DataFrame")
    return list(df.select_dtypes(include=['category', 'object']))


def get_numeric_columns(df):
    """
    Returns numerical columns from pandas dataframe

    Parameters
    ----------
    df : Pandas Dataframe

    Returns
    -------
    Python list
    """
    if df is None:
        raise ValueError(
            "The parameter 'df' must be assigned a non-nil reference to a Pandas DataFrame")
    return list(df.select_dtypes(exclude=['category', 'object']))

def describe_tcat_by_fnum(df, f_num, t_cat, verbose=False):
    grouped = df.groupby(t_cat)
    grouped_y = grouped[f_num].describe()
    if verbose:
        print(grouped_y)
    return grouped_y

def relative_freq_fcat_by_tcat(df, f_cat, t_cat, verbose=False):
    grouped = df.groupby(f_cat)
    grouped_x = grouped[t_cat].value_counts(normalize=True)
    if verbose:
        print(grouped_x)
    return grouped_x

def get_quantiles(df, cols):
    return df[cols].quantile(q=[.01, .25, .5, .75, .95, .99])

def get_leq_quantile(df, col, q):
    return df[(df[col] <= df[col].quantile(q=q))]

def get_geq_quantile(df, col, q):
    return df[(df[col] >= df[col].quantile(q=q))]

def get_between_quantiles(df, col, qs):
    lower_q = min(qs)
    upper_q = max(qs)
    return df[(df[col] >= df[col].quantile(q=lower_q)) & (df[col] <= df[col].quantile(q=upper_q))]

def plot_categorical__reg(df, target_col, min_card=2, max_card=12, height=5, ascpect=2, rotation=45, color='grey', kind='bar'):
    """
    Plots categorical features vs. numerical target

    Parameters
    ----------
    df: Pandas Dataframe
    traget_col: Target variable, dependent variable
    min_card: Minimum cardinality of categorical feature
    max_card: Maximum cardinality of categorical feature
    """
    cat_columns = get_categorical_columns(df)
    for col in sorted(cat_columns):
        if (df[col].nunique() >= min_card) and (df[col].nunique() < max_card):
            sns.catplot(x=col, y=target_col, data=df, kind=kind,
                        color=color, height=height, aspect=ascpect)
            plt.xticks(rotation=rotation)
            plt.show()
            plt.close()
    return None

def plot_numerical_columns_reg(df, target_col, alpha=0.5, color='grey'):
    """
    Plots numerical features vs. numerical target

    Parameters
    ----------
    df : Pandas Dataframe
    target_col : Target variable, dependent variable
    """
    num_columns = get_numeric_columns(df)
    for col in sorted(num_columns):
        if col != target_col:
            sns.lmplot(x=col, y=target_col, data=df,
                       scatter_kws=dict(alpha=alpha, color=color))
            plt.show()
            plt.close()




def plot_correlation_heatmap(data=None, vmax=1, annot=True, corr_type='pearson', figsize=(12, 12)):
    """
    Plots correlations on a heatmap
    """
    if data is None:
        raise ValueError(
            "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
    # Compute the correlation matrix
    corr = data.corr(corr_type)
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    fig, axes = plt.subplots(figsize=figsize)
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=vmax, annot=annot, square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=axes)
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes=None,
                            normalize=False,
                            title=None,
                            cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    labels_is_string = False
    uniq_labels = unique_labels(y_true, y_pred)
    for val in uniq_labels:
        if isinstance(val, str):
            labels_is_string = True
            break
    if labels_is_string:
        classes = uniq_labels
    else:
        classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def lowess_scatter(data, x, y, jitter=0.0, skip_lowess=False):
    if skip_lowess:
        fit = np.polyfit(data[x], data[y], 1)
        line_x = np.linspace(data[x].min(), data[x].max(), 10)
        line = np.poly1d(fit)
        line_y = list(map(line, line_x))
    else:
        lowess = sm.nonparametric.lowess(data[y], data[x], frac=.3)
        line_x = list(zip(*lowess))[0]
        line_y = list(zip(*lowess))[1]

    figure = plt.figure(figsize=(10, 6))
    axes = figure.add_subplot(1, 1, 1)
    xs = data[x]
    if jitter > 0.0:
        xs = data[x] + stats.norm.rvs(0, 0.5, data[x].size)

    axes.scatter(xs, data[y], marker="o", color="steelblue", alpha=0.5)
    axes.plot(line_x, line_y, color="DarkRed")

    title = "Plot of {0} v. {1}".format(x, y)

    if not skip_lowess:
        title += " with LOESS"
    axes.set_title(title)
    axes.set_xlabel(x)
    axes.set_ylabel(y)

    plt.show()
    plt.close()

def plot_scatter_by_groups(df, x_col, y_col, group_by_col, colors=None, alpha=0.75):
    labels = df[group_by_col].unique()
    num_labels = np.arange(1, len(labels)+1)
    fig, ax = plt.subplots()
    for idx, label in zip(num_labels, labels):
        indices_to_keep = df[group_by_col] == label
        y = df.loc[indices_to_keep, y_col]
        if x_col == 'index':
            x = df.index[indices_to_keep]
        else:
            x = df.loc[indices_to_keep, x_col]
        ax.scatter(x, y, label=label, alpha=alpha)
    plt.show()
    plt.close()
