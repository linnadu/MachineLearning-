import pandas as pd
from random import random
from random import seed
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from importlib import import_module
import sklearn
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
# regression
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
# classifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.linear_model import ElasticNetCV, ElasticNet
def mean_absolute_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred))) * 100

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

###################################################################################################################


'''
corrlation plot
'''
#corr = corrlationPlot(df)
def corrlationPlot(df):
    covMatrix = pd.DataFrame.cov(df)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(corr, annot=True, fmt='g')
    mask = np.zeros_like(data.corr())
    mask[np.triu_indices_from(mask)] = 1
    sns.heatmap(corr, mask= mask, ax= ax, annot= True)
    plt.show()
    return corr

#important_features, drop_features = ElasticNet(X_train, y_train)
def ElasticNet(X_train, y_train):
    cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True, 
                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, 
                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
    cv_model.fit(X_train, y_train)
    print('Optimal alpha: %.8f'%cv_model.alpha_)
    print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
    print('Number of iterations %d'%cv_model.n_iter_)
    model = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha = cv_model.alpha_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)
    model.fit(X_train, y_train)
    print(r2_score(y_train, model.predict(X_train)))
    feature_importance = pd.Series(index = X_train.columns, data = np.abs(model.coef_))
    important_features = feature_importance[feature_importance>0]
    drop_features = feature_importance[feature_importance<=0]
    n_selected_features = important_features.shape[0]
    print('{0:d} features, reduction of {1:2.2f}%'.format(
    n_selected_features,(1-n_selected_features/len(feature_importance))*100))
    feature_importance.sort_values().tail(50).plot(kind = 'bar', figsize = (18,6))
    return important_features, drop_features


def recursionFeatureSelection(X_train, y_train):
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(X_train, y_train)
    ranking = rfe.ranking_.reshape(digits.images[0].shape)
    # Plot pixel ranking
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Ranking of pixels with RFE")
    plt.show()


'''
Spot check Algorithms
'''
Rmodels = []
Rmodels.append(('LR', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)))
Rmodels.append(('LGBM', LGBMRegressor()))
Rmodels.append(('XGB', XGBRegressor()))
#Rmodels.append(('CatBoost', CatBoostRegressor()))
Rmodels.append(('SGD', SGDRegressor()))
Rmodels.append(('KernelRidge', KernelRidge()))
Rmodels.append(('ElasticNet', ElasticNet()))
Rmodels.append(('BayesianRidge', BayesianRidge()))
Rmodels.append(('GradientBoosting', GradientBoostingRegressor()))
Rmodels.append(('SVR', SVR(gamma='auto'))) # kernel = linear, svr
Rmodels.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
Rmodels.append(('KNN', KNeighborsRegressor()))    # kneighbor
Rmodels.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees

 # decision tree
 # Gradient boosting
Cmodels = []
Cmodels.append(('Logistic', LogisticRegression()))
Cmodels.append(('SVM', SVC(gamma='auto')))
Cmodels.append(('GaussianNB', GaussianNB()))
Cmodels.append(('MultinominalNB', MultinomialNB()))
Cmodels.append(('SGD', SGDClassifier()))
Cmodels.append(('DecisionTree', DecisionTreeClassifier()))
Cmodels.append(('RF', RandomForestClassifier()))
Cmodels.append(('GradientBoosting', GradientBoostingClassifier()))
Cmodels.append(('LGBM', LGBMClassifier()))
Cmodels.append(('XGB', XGBClassifier()))
'''
preprocessing and encoding
'''
def Convert_categorical_lable(arr):
    le = preprocessing.LabelEncoder()
    le.fit(arr)
    return le.classes_, le.transform(arr)
categorical_encoder = OneHotEncoder(handle_unknown='ignore')

'''
Evaluate each model in cross validataion 
'''
def Train_Models_TimeSplit(X_train, y_train, modellist, score_fuction = make_scorer(mean_absolute_error)):
    results = {}
    for name, model in modellist:
        print(name, model)
        tscv = TimeSeriesSplit(n_splits=2) # TimeSeries Cross validation
        cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring= score_fuction)
        results[name] = cv_results
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    # Compare Algorithms
    results = pd.DataFrame.from_dict(results, orient='index', columns=['mean', 'std'])
    return results #

'''
feature importance
'''
'''
r = permutation_importance(model, X_val, y_val,n_repeats=30, random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{diabetes.feature_names[i]:<8}"
             f"{r.importances_mean[i]:.3f}"
               f" +/- {r.importances_std[i]:.3f}")

'''
#randomforest feature importance 

'''
performance metrics
'''

def regression_Prediction_Matrics(y_true, y_pred):
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))        
    

'''
Grid Searching Hyperparameters
'''
'''
from sklearn.model_selection import GridSearchCV
model = RandomForestRegressor()
param_search = { 
    'n_estimators': [20, 50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [i for i in range(5,15)]
}
tscv = TimeSeriesSplit(n_splits=10)
gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
gsearch.fit(X_train, y_train)
best_score = gsearch.best_score_
best_model = gsearch.best_estimator_
'''
  
######################################################################
# examples
######################################################################
path='/Users/dulinna/LDProjects/lstm/'
#dataD =pd.read_csv(path +"/NasdaqDaily.csv", index_col = 0)
#dataW =pd.read_csv(path +"/NasdaqWeekly.csv", index_col = 0)
#dataM =pd.read_csv(path +"/NasdaqMonthly.csv", index_col = 0)
filename = 'bunkrupt.csv'
data = pd.read_csv(path + filename) #, index_col = 0)
summary = data.describe().T # ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']


'''
feature selection

'''

corr = corrlationPlot(df)
#Elastic - Net
#Removing features with low variance
#feature important by regularization
important_features, drop_features = ElasticNet(X_train, y_train)
#feature important by recursion selection (subset of features)



data = data
y = data['Bankrupt?']#.pct_change().dropna()
# daily #.mean(); .std(); .corr(); .cov; .shift(); .diff(); .dropna() ;
X = data.loc[:, data.columns != 'Bankrupt?']#.pct_change().dropna()  
y.plot.hist(bins = 60)
#cumulative return 
(y+1).cumprod()


#random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, train_size=0.2, random_state=None, shuffle=None, stratify=None)

X_train, X_test, y_train, y_test = X['2000-01-03':'2020-01-01'], X['2020-01-01':], y['2000-01-03':'2020-01-01'], y['2020-01-01':]
#result = Train_Models_TimeSplit(X_train, y_train, Rmodels, score_fuction = make_scorer(mean_absolute_error))

#Timeseries split
'''
tscv = TimeSeriesSplit(n_splits=5) 
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index] 
'''

make_pipeline(StandardScaler(), GaussianNB(priors=None))
