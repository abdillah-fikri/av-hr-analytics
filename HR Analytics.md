---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
    metadata:
      interpreter:
        hash: 9147bcb9e0785203a659ab3390718fd781c9994811db246717fd6ffdcf1dd807
    name: 'Python 3.8.5 64-bit (''ds_env'': conda)'
---

```python colab={} colab_type="code" executionInfo={"elapsed": 1317, "status": "ok", "timestamp": 1600238275690, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="TbArW04SJ39R"
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style='darkgrid', palette='muted')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 4082, "status": "ok", "timestamp": 1600238278473, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="EUlk_yifP10z" outputId="d4aca1e5-42d3-4479-a383-8566478a9a27"
pip install seaborn
```

```python colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" executionInfo={"elapsed": 5978, "status": "ok", "timestamp": 1600238280386, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="JnqCNbNGQJD2" outputId="f269f04d-9c9a-4433-d8dc-6f23e49ea0f8"
pip install category_encoders
```

```python colab={} colab_type="code" executionInfo={"elapsed": 5963, "status": "ok", "timestamp": 1600238280387, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="zjkT3K2eJ39W"
train = pd.read_csv('train_LZdllcl.csv')
test = pd.read_csv('test_2umaH9m.csv')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 292} colab_type="code" executionInfo={"elapsed": 5950, "status": "ok", "timestamp": 1600238280388, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Naz2sONsJ39a" outputId="d5c2057a-61d9-47fb-c379-b456c2e55e33"
train.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 5931, "status": "ok", "timestamp": 1600238280389, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="GN5JDGTAJ39f" outputId="779f0672-5a8e-4337-c9d4-8cbf58539ae3"
train.shape, test.shape
```

```python colab={"base_uri": "https://localhost:8080/", "height": 374} colab_type="code" executionInfo={"elapsed": 5913, "status": "ok", "timestamp": 1600238280390, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="HmErnJDUJ39i" outputId="2f59e008-99b6-4836-a6c9-4b3c3b03031f" tags=[]
train.info()
```

```python colab={} colab_type="code" executionInfo={"elapsed": 5898, "status": "ok", "timestamp": 1600238280390, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="63GlwwwWJ39m"
num_cont = [col for col in train.columns if (train[col].dtype in ['int64', 'float64']) 
            and (train[col].nunique() > 10)
            and (col not in ['employee_id', 'is_promoted'])]

num_disc = [col for col in train.columns if (train[col].dtype in ['int64', 'float64']) 
            and (train[col].nunique() <= 10)
            and (col not in ['employee_id', 'is_promoted'])]

cat_cols = [col for col in train.columns if (train[col].dtype in ['object'])
            and (col not in ['employee_id', 'is_promoted'])]
target = ['is_promoted']
```

```python colab={"base_uri": "https://localhost:8080/", "height": 272} colab_type="code" executionInfo={"elapsed": 5882, "status": "ok", "timestamp": 1600238280391, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="dqw50_7oJ39q" outputId="56fa661c-78ea-4d17-a02e-edd9ed2f50e2"
train.isna().sum()/train.shape[0]*100
```

<!-- #region colab_type="text" id="URCLrTHDJ39u" -->
# Data Undestanding
<!-- #endregion -->

<!-- #region colab_type="text" id="HkTBoB3NJ39u" -->
## Target Distribution
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 356} colab_type="code" executionInfo={"elapsed": 5862, "status": "ok", "timestamp": 1600238280392, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="N-8Ded71J39v" outputId="1eb6920a-2eeb-4ace-af1d-0f93f78b6264"
sns.countplot(train[target[0]])
```

<!-- #region colab_type="text" id="23ycsFOgJ39y" -->
## Numerical Features Distribution
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} colab_type="code" executionInfo={"elapsed": 7898, "status": "ok", "timestamp": 1600238282448, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="TFs7RTK5J39z" outputId="62419ae6-5999-4a19-91e2-79339cf2169a"
plt.figure(figsize=(20,4))
for i, col in enumerate(num_cont):
    plt.subplot(1,3,i+1)
    sns.histplot(train[col])
plt.tight_layout(pad=2.0)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 401} colab_type="code" executionInfo={"elapsed": 8577, "status": "ok", "timestamp": 1600238283145, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="FAiNdXuHJ392" outputId="72a0c7c6-1b68-41fe-811c-6ea3a8eb4095"
plt.figure(figsize=(20,4))
for i, col in enumerate(num_cont):
    plt.subplot(1,3,i+1)
    sns.boxplot(train[col])
plt.tight_layout(pad=2.0)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 435} colab_type="code" executionInfo={"elapsed": 9141, "status": "ok", "timestamp": 1600238283729, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="VnyyR2GjJ396" outputId="ca7273d4-4b4f-44c3-cd9a-8dcac6d30a28"
plt.figure(figsize=(20,4))
for i, col in enumerate(num_disc):
    plt.subplot(1,4,i+1)
    sns.countplot(train[col])
plt.tight_layout(pad=2.0)
```

<!-- #region colab_type="text" id="QoqLaWRzJ399" -->
## Categorical Features Distribution
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 9127, "status": "ok", "timestamp": 1600238283732, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="zB-oFGclJ39-" outputId="c4a98320-4bae-4b31-b773-1c0d2ed95aaf"
len(cat_cols)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 587} colab_type="code" executionInfo={"elapsed": 11312, "status": "ok", "timestamp": 1600238285937, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Se7fr7vdJ3-B" outputId="28c1721e-5cf0-45f7-b087-8da57fe563e4"
plt.figure(figsize=(20,8))
for i, col in enumerate(cat_cols):
    plt.subplot(2,3,i+1)
    sns.countplot(data=train, x=col)
    if col in ['department', 'region']:
        plt.xticks(rotation=90)
plt.tight_layout(pad=1.0)
```

<!-- #region colab_type="text" id="d4UpkbtdJ3-F" -->
## Bivariate Analysis
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 297} colab_type="code" executionInfo={"elapsed": 11297, "status": "ok", "timestamp": 1600238285940, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="ohE8Rp6oJ3-F" outputId="ccbaefec-4c3f-488c-f0ed-365640c28122"
corr = train[num_cont + num_disc + target].corr()
corr
```

```python colab={"base_uri": "https://localhost:8080/", "height": 612} colab_type="code" executionInfo={"elapsed": 11280, "status": "ok", "timestamp": 1600238285941, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Fm62ZrgTJ3-I" outputId="bc4f3fa3-1467-4d9a-a251-dc8d3ac1125a"
plt.figure(figsize=(10,8))
sns.heatmap(corr , annot=True, linewidths=.5, fmt= '.2f')
```

```python colab={"base_uri": "https://localhost:8080/", "height": 351} colab_type="code" executionInfo={"elapsed": 12325, "status": "ok", "timestamp": 1600238287006, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="UB8nv4syJ3-L" outputId="75883beb-b757-4517-a954-4458cfe2c0c9"
plt.figure(figsize=(10,5))
for i, col in enumerate(num_cont):
    plt.subplot(1,3,i+1)
    sns.boxplot(data=train, x=target[0], y=col)
plt.tight_layout(pad=2.0)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 455} colab_type="code" executionInfo={"elapsed": 12311, "status": "ok", "timestamp": 1600238287009, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="F-lOeOMhJ3-O" outputId="74b3f832-5dfc-42c6-8487-d60c58e5750e"
plt.figure(figsize=(20,4))
for i, col in enumerate(num_disc):
    plt.subplot(1,4,i+1)
    sns.countplot(train[col], hue=train[target[0]])
plt.tight_layout(pad=1.0)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 777} colab_type="code" executionInfo={"elapsed": 14471, "status": "ok", "timestamp": 1600238289187, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="-rcxB-hfJ3-S" outputId="9935db7f-3d19-4804-ad25-b44e279a525e"
plt.figure(figsize=(20,8))
for i, col in enumerate(cat_cols):
    plt.subplot(2,3,i+1)
    sns.countplot(train[col], hue=train[target[0]])
    if col in ['department', 'region']:
        plt.xticks(rotation=90)
plt.tight_layout(pad=1.0)
```

<!-- #region colab_type="text" id="kZIeaohjJ3-V" -->
# Preprocessing
<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 14460, "status": "ok", "timestamp": 1600238289188, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="1s7eM_qIJ3-V"
X = train.drop(columns=['employee_id', 'is_promoted'])
y = train['is_promoted']
X_test = test.drop(columns=['employee_id'])
```

<!-- #region colab_type="text" id="-_CbGViAJ3-Y" -->
## Missing Value
<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 14450, "status": "ok", "timestamp": 1600238289189, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="ywX-ORqKJ3-Z"
from sklearn.impute import SimpleImputer
```

```python colab={"base_uri": "https://localhost:8080/", "height": 0} colab_type="code" executionInfo={"elapsed": 14442, "status": "ok", "timestamp": 1600238289190, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="xt-8cIF-J3-c" outputId="30988d0d-4f9d-4fb1-e32b-9ef3bf02689f"
na_col = pd.DataFrame(X.isna().sum()) / X.shape[0]*100
na_col.columns = ['NA Train']
na_col['NA Test'] = X_test.isna().sum().values / X_test.shape[0]*100
round(na_col.sort_values(by='NA Train', ascending=False), 2)
```

```python colab={} colab_type="code" executionInfo={"elapsed": 15810, "status": "ok", "timestamp": 1600238290567, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="eOqJW0_CJ3-f"
imputer = SimpleImputer(strategy='most_frequent')
X['education'] =  imputer.fit_transform(X['education'].values.reshape(-1, 1))
X_test['education'] =  imputer.transform(X_test['education'].values.reshape(-1, 1))
```

```python colab={} colab_type="code" executionInfo={"elapsed": 15803, "status": "ok", "timestamp": 1600238290568, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="XB9KHD_GJ3-h"
X['previous_year_rating'] =  imputer.fit_transform(X['previous_year_rating'].values.reshape(-1, 1))
X_test['previous_year_rating'] =  imputer.transform(X_test['previous_year_rating'].values.reshape(-1, 1))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 0} colab_type="code" executionInfo={"elapsed": 15795, "status": "ok", "timestamp": 1600238290569, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="n-MetUc6J3-k" outputId="2aebf875-60a5-4fda-f68a-ee6a47ff82b4"
na_col = pd.DataFrame(X.isna().sum()) / X.shape[0]*100
na_col.columns = ['NA Train']
na_col['NA Test'] = X_test.isna().sum().values / X_test.shape[0]*100
round(na_col.sort_values(by='NA Train', ascending=False), 2)
```

<!-- #region colab_type="text" id="TWWaJ_w5J3-q" -->
## Categorical Encoding
<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 15786, "status": "ok", "timestamp": 1600238290570, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="CcB-Lcj8J3-r"
edu_map = {'Below Secondary': 1, 'Bachelor\'s': 2, 'Master\'s & above': 3}
gen_map = {'f': 0, 'm': 1}

for col, mapping in zip(['education', 'gender'], [edu_map, gen_map]):
    X[col] = X[col].map(mapping)
    X_test[col] = X_test[col].map(mapping)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 0} colab_type="code" executionInfo={"elapsed": 15777, "status": "ok", "timestamp": 1600238290571, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="9qw3-KZnJ3-v" outputId="118c6bfb-2e05-4710-e199-9fd4fa098caf"
import category_encoders as ce
onehot = ce.OneHotEncoder(cols=[col for col in X.columns if X[col].dtype == 'object'], use_cat_names=True)
```

```python colab={} colab_type="code" executionInfo={"elapsed": 15766, "status": "ok", "timestamp": 1600238290571, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="F1LJ3LNhJ3-y"
X = onehot.fit_transform(X)
X_test = onehot.transform(X_test)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 0} colab_type="code" executionInfo={"elapsed": 16380, "status": "ok", "timestamp": 1600238291198, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="sfGAL3u8J3-0" outputId="32886cd2-9f38-47a2-e078-45cc8f1d2443"
X.head()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 0} colab_type="code" executionInfo={"elapsed": 16373, "status": "ok", "timestamp": 1600238291200, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="ELP_SqRLJ3-3" outputId="c425e24f-1a59-4002-941e-7296b3197540"
X.shape, X_test.shape
```

<!-- #region colab_type="text" id="XaE0Pr5tJ3-7" -->
# Modeling
<!-- #endregion -->

```python colab={} colab_type="code" executionInfo={"elapsed": 16362, "status": "ok", "timestamp": 1600238291202, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="XFKww0_jJ3-7"
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
from xgboost import XGBClassifier
```

```python colab={} colab_type="code" executionInfo={"elapsed": 16354, "status": "ok", "timestamp": 1600238291204, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="G3m8MseIJ3--"
def model_eval(model, X, y, scoring_='roc_auc', cv_=5):
    #Fit the algorithm on the data
    model.fit(X, y)
        
    #Predict training set:
    pred = model.predict(X)
    predprob = model.predict_proba(X)[:,1]

    cv_score = cross_val_score(model, X, y, cv=cv_, scoring=scoring_)
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y.values, pred))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y.values, predprob))
    
    print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" executionInfo={"elapsed": 18857, "status": "ok", "timestamp": 1600238293722, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="nAGVdkWnJ3_B" outputId="819fb33a-99d4-4395-9e24-f204e29ae518" tags=[]
base_model = XGBClassifier(seed=14, tree_method='gpu_hist')
model_eval(base_model, X, y)
```

```python colab={} colab_type="code" id="rb4HEZWqJ3_E" outputId="0ec6ad4f-1ee5-4385-fc2e-3537fde8e95b"
# Step 1: Get initial fix learning_rate and n_estimators
test1 = {'n_estimators':range(20,101,10)}
grid1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
                                               max_depth=5, 
                                               min_child_weight=1, 
                                               gamma=0, 
                                               subsample=0.8, 
                                               colsample_bytree=0.8,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=1, 
                                               seed=14), 
                                               param_grid = test1, 
                                               scoring='roc_auc',
                                               n_jobs=-1,iid=False, cv=5)
grid1.fit(X, y)
grid1.best_params_, grid1.best_score_
```

```python colab={} colab_type="code" id="f1P87mU3J3_H" outputId="0c54c988-e298-4510-90c8-dcf31a30831d" tags=[]
# Step 2: Tune max_depth and min_child_weight
test2 = {'max_depth': range(3,10,2),
         'min_child_weight': range(1,6,2)}
grid2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                               n_estimators=100,
                                               max_depth=5, 
                                               min_child_weight=1, 
                                               gamma=0, 
                                               subsample=0.8, 
                                               colsample_bytree=0.8,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=1, 
                                               seed=14), 
                                               param_grid = test2, 
                                               scoring='roc_auc',
                                               n_jobs=-1,iid=False, cv=5)
grid2.fit(X, y)
grid2.best_params_, grid2.best_score_
```

```python colab={} colab_type="code" id="WPSbHLsaJ3_J" outputId="4d5eccea-5109-4149-bbf5-8c9ad55447d1"
# Step 2b: Tune max_depth and min_child_weight
test2 = {'max_depth': [6,7,8],
         'min_child_weight': [2,3,4]}
grid2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                               n_estimators=100,
                                               max_depth=5, 
                                               min_child_weight=1, 
                                               gamma=0, 
                                               subsample=0.8, 
                                               colsample_bytree=0.8,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=1, 
                                               seed=14), 
                                               param_grid = test2, 
                                               scoring='roc_auc',
                                               n_jobs=-1,iid=False, cv=5)
grid2.fit(X, y)
grid2.best_params_, grid2.best_score_
```

```python colab={} colab_type="code" id="zpgAr6mdJ3_M" outputId="1efe51c0-1901-433a-e075-4638b219bd6d"
# Step 3: Tune gamma
test3 = {'gamma': [i/10.0 for i in range(0,5)]}
grid3 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                               n_estimators=100,
                                               max_depth=7, 
                                               min_child_weight=4, 
                                               gamma=0, 
                                               subsample=0.8, 
                                               colsample_bytree=0.8,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=1, 
                                               seed=14), 
                                               param_grid = test3, 
                                               scoring='roc_auc',
                                               n_jobs=-1,iid=False, cv=5)
grid3.fit(X, y)
grid3.best_params_, grid3.best_score_
```

```python colab={} colab_type="code" id="4zyAdYxwJ3_P" outputId="0e01f1a4-b788-47a4-913f-0e7f6a429f05"
# Step 4: Tune subsample and colsample_bytree
test4 = {'subsample':[i/10.0 for i in range(6,10)],
         'colsample_bytree':[i/10.0 for i in range(6,10)]}
grid4 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                               n_estimators=100,
                                               max_depth=7, 
                                               min_child_weight=4, 
                                               gamma=0, 
                                               subsample=0.8, 
                                               colsample_bytree=0.8,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=1, 
                                               seed=14), 
                                               param_grid = test4, 
                                               scoring='roc_auc',
                                               n_jobs=-1,iid=False, cv=5)
grid4.fit(X, y)
grid4.best_params_, grid4.best_score_
```

```python colab={} colab_type="code" id="KpaF8EJ3J3_S" outputId="08602290-0bc3-48e8-ef5a-781b532b8cf5"
[i/10 for i in range(1,11)]
```

```python colab={} colab_type="code" id="cjpYlTEbJ3_V" outputId="4ef470f0-74d1-48ca-b8be-8b3e29be4612"
# Step 5: Tuning scale_pos_weight
test5 = {'scale_pos_weight': [i/10 for i in range(1,11)]}
grid5 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                               n_estimators=100,
                                               max_depth=7, 
                                               min_child_weight=4, 
                                               gamma=0, 
                                               subsample=0.9, 
                                               colsample_bytree=0.6,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=0.9, 
                                               seed=14), 
                                               param_grid = test5, 
                                               scoring='roc_auc',
                                               n_jobs=-1,iid=False, cv=5)
grid5.fit(X, y)
grid5.best_params_, grid5.best_score_
```

```python colab={} colab_type="code" id="eIZPSPWlJ3_Y" outputId="34d00046-c810-4e35-cffe-474f7f5cafb4"
# Step 6: Tuning n_estimators and learning_rate
test6 = {'learning_rate': [0.05],
         'n_estimators': range(200,1001,200)}
grid6 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                               n_estimators=100,
                                               max_depth=7, 
                                               min_child_weight=4, 
                                               gamma=0, 
                                               subsample=0.9, 
                                               colsample_bytree=0.6,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=0.9, 
                                               seed=14), 
                                               param_grid = test6, 
                                               scoring='roc_auc',
                                               n_jobs=-1,iid=False, cv=5)
grid6.fit(X, y)
grid6.best_params_, grid6.best_score_
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 511663, "status": "ok", "timestamp": 1600246498809, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="26fTuEx6J3_g" outputId="956fe324-ccc8-42b0-9fd9-4b92cf51844b"
# Step 6b: Tuning n_estimators and learning_rate
test6 = {'learning_rate': [0.005],
         'n_estimators': range(2550,2571,5)}
grid6 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,
                                               n_estimators=100,
                                               max_depth=7, 
                                               min_child_weight=4, 
                                               gamma=0,
                                               subsample=0.9, 
                                               colsample_bytree=0.6,
                                               objective= 'binary:logistic', 
                                               scale_pos_weight=0.9, 
                                               seed=14,
                                               tree_method='gpu_hist'), 
                                               param_grid = test6, 
                                               scoring='roc_auc',
                                               n_jobs=-1, cv=5)
grid6.fit(X, y)
grid6.best_params_, grid6.best_score_
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" executionInfo={"elapsed": 39105, "status": "ok", "timestamp": 1600238429081, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="wQ4Na2AvM78W" outputId="076d2140-e6ca-4410-9f96-f8cc9704a67c"
xgb_gpu_f1 = XGBClassifier(learning_rate=0.01,
                           n_estimators=1250,
                           max_depth=7, 
                           min_child_weight=4,
                           gamma=0,
                           subsample=0.9,
                           colsample_bytree=0.6,
                           objective= 'binary:logistic', 
                           scale_pos_weight=0.9,
                           seed=14,
                           tree_method='gpu_hist')
model_eval(xgb_gpu_f1, X, y)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" executionInfo={"elapsed": 76444, "status": "ok", "timestamp": 1600241064100, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="fw6pik28J3_m" outputId="e82d9f25-112d-4386-f2fd-ea601fad0b7e" tags=[]
xgb_gpu_f2 = XGBClassifier(learning_rate=0.005,
                           n_estimators=2560,
                           max_depth=7, 
                           min_child_weight=4,
                           gamma=0,
                           subsample=0.9,
                           colsample_bytree=0.6,
                           objective= 'binary:logistic', 
                           scale_pos_weight=0.9,
                           seed=14,
                           tree_method='gpu_hist')
model_eval(xgb_gpu_f2, X, y)
```

```python colab={} colab_type="code" executionInfo={"elapsed": 1242, "status": "ok", "timestamp": 1600241094248, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="NSvs4NQiJ3_q"
import xgboost as xgb
```

```python colab={"base_uri": "https://localhost:8080/", "height": 916} colab_type="code" executionInfo={"elapsed": 3179, "status": "ok", "timestamp": 1600241096986, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="5OFjfIqVJ3_t" outputId="86efc3e3-2aec-47d1-8924-b0e0caaa551c"
plt.figure(figsize=(20,15))
xgb.plot_importance(xgb_gpu_f2, ax=plt.gca())
```

```python colab={} colab_type="code" executionInfo={"elapsed": 1236, "status": "ok", "timestamp": 1600241113798, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="yXHQkTXqJ3_w"
pred_test = xgb_gpu_f2.predict(X_test)
output = pd.DataFrame({'employee_id': test['employee_id'],
                       'is_promoted': pred_test})
output.to_csv('submission.csv', index=False)
```

```python colab={} colab_type="code" id="uNAgDuVdJ3_y"

```
