import pandas as pd
import numpy as np
from collections import Counter 
import matplotlib.pyplot as plt
import datetime as dt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics



path='/Users/dulinna/LDProjects/pickingstock/RobinHood_Test'
start = dt.date(2016,8, 16)
end = dt.date(2017,8, 17)
equity_value_data = pd.read_csv(path +"/equity_value_data.csv")
features_data = pd.read_csv(path +"/features_data.csv")

print('(a) The percentage of Users have churned in the data provided')
# parse the time step to find the date 
equity_value_data['date'] = equity_value_data['timestamp'].apply (lambda x : x.split('T')[0])
equity_value_data.drop(columns = ['timestamp'], inplace = True)
equity_value_groupby = equity_value_data.groupby(['user_id'])
'''
calculate the gap between the data collection data and the last date the user is acctive
Asume that when the user acount is not in the dataset, the user is churned
also calculate the equity volatilty, portfolio changes for each user during the time period
'''
all = pd.DataFrame(columns = ['user_id','fist_day', 'last_day', 'days_gap', 'close_equity', 'first_last_day_changes_pct', 'equity_volatility'])
for k, v in equity_value_groupby:
    v.sort_values(by = ['date'])
    firstday = min(v['date'])
    y,m,d = max(v['date']).split('-')
    lastday = dt.date(int(y), int(m), int(d))
    gap = (end - lastday).days
    ininital_euity = v[v.date == min(v['date'])   ]['close_equity'].values[0]
    close_equity = v[v.date == max(v['date'])   ]['close_equity'].values[0]
    equity_changes_pct = (close_equity - ininital_euity)%ininital_euity
    vol = np.std( np.log(v['close_equity']/v['close_equity'].shift(-1)) )*252**0.5
    all = (all.append({'user_id': k,'fist_day': firstday, 'last_day':lastday, 'close_equity':close_equity,  'days_gap': gap,
                       'first_last_day_changes_pct':equity_changes_pct, 'equity_volatility': vol },
                       ignore_index = True))  
aa = all
churn = all[all.days_gap >= 28]
no_churn = all[all.days_gap < 28]
print("the churn rate in % is")
churnedrate = len(churn)/len(all) * 100
print(churnedrate)

print('b) build a classifer')

data = all.merge(features_data, how = 'left', on ='user_id')
data['acount_changes'] = (data.close_equity - data.first_deposit_amount)/data.first_deposit_amount * 100
data['Churn_or_not'] = [1 if i >= 28 else 0 for i in data['days_gap']]
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace = True)

y = data['Churn_or_not']
reduced = ['liquidity_needs', 'instrument_type_first_traded', 'platform', 'time_horizon']
data.drop(columns = ['user_id', 'fist_day', 'last_day', 'days_gap', 'close_equity', 'Churn_or_not'], inplace = True)
X = data
categorical_data  = ['risk_tolerance','investment_experience', 'liquidity_needs', 'instrument_type_first_traded', 'platform', 'time_horizon']
numerical_data = ['time_spent','acount_changes', 'equity_volatility', 'first_last_day_changes_pct', 'first_deposit_amount']

# deal with categorial data
col_trans = make_column_transformer(
                        (OneHotEncoder(),categorical_data),
                        remainder = "passthrough"
                        )




rf_classifier = RandomForestClassifier()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.8, random_state=50, shuffle=True, stratify=None)
X_train = X_train.fillna('na')
X_test = X_test.fillna('na')

## upscale sampling
X_train_y = X_train
X_train_y['y'] = y_train
minority = X_train_y[X_train_y.y ==1]
majority = X_train_y[X_train_y.y ==0]
minority_upsampled  = resample(minority,
                               replace = True,
                               n_samples = majority.shape[0],
                               random_state = 123)
df_upsampled = pd.concat([minority_upsampled, majority])
y_train = df_upsampled['y']
X_train = df_upsampled.drop(columns = ['y'])

# train with logistic regression

logistic_classifier = LogisticRegression()
pipe_logit = make_pipeline(col_trans, logistic_classifier)
pipe_logit.fit(X_train, y_train)
y_pred_logist = pipe_logit.predict(x_test)
score = pipe_logit.score(X_test, y_test)
print(score)
'''
#train model with simple RF
#
seed = 50
rf_classifier = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')
pipe = make_pipeline(col_trans, rf_classifier)
pipe.fit(X_train, y_train)


# grid search and cross validation
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 300, num = 50)],
               'max_features': ['auto', 'log2'],
               'max_depth': [3, 5],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 4, 10],
               'max_leaf_nodes': [None] + list(np.linspace(10, 50, 100).astype(int)),
               'bootstrap': [True, False], 
               'n_jobs': [-1]
            
              }
rf_tune = RandomForestClassifier(oob_score=True, n_jobs = -1)
random_rf = RandomizedSearchCV(
                estimator = rf_tune,
                param_distributions = param_grid,
                verbose = 2,
                random_state=seed,
                cv=3,
                scoring='roc_auc')

pipe_random = make_pipeline(col_trans, random_rf)
pipe_random.fit(X_train, y_train)
# cross validation
random_rf.best_params_
best_model = random_rf.best_estimator_
pipe_best_model = make_pipeline(col_trans, best_model)
pipe_best_model.fit(X_train, y_train)
y_pred_best_model = pipe_best_model.predict(X_test)
train_rf_predictions = pipe_best_model.predict(X_train)
train_rf_probs = pipe_best_model.predict_proba(X_train)[:, 1]
rf_probs = pipe_best_model.predict_proba(X_test)[:, 1]
accuracy_score(y_test, y_pred)
print(f"The accuracy of the model is {round(accuracy_score(y_test,y_pred_best_model ),3)*100} %")
print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_rf_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, rf_probs)}')


'''
y_pred = pipe.predict(X_test)




'''https://towardsdatascience.com/my-random-forest-classifier-cheat-sheet-in-python-fedb84f8cf4f'''
