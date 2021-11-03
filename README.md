# E-commerce customer Churn Prediction
## Introduction
### What is the churn rate?
The churn rate, also known as the rate of attrition or customer churn, is the rate at which customers stop doing business with an entity. It is most commonly expressed as the percentage of service subscribers who discontinue their subscriptions within a given time period. It is also the rate at which employees leave their jobs within a certain period. For a company to expand its clientele, its growth rate (measured by the number of new customers) must exceed its churn rate.

### The Importance of the Customer Churn Rate
1. Firstly, its indication of lost customers directly ties into lost revenue. When a company loses customers, it negatively affects its bottom line. In this way, a high churn limits a business’s growth potential. Additionally, a significant churn rate always incurs financial difficulties of some degree.

2. Another reason it is so important to study and improve customer churn is that it is far more expensive to acquire new customers than it is to retain existing ones. In fact, acquiring a new customer costs 5 times more than retaining an existing one. Therefore, when a business loses existing customers, it will have to grapple with both the loss of revenue and business opportunities, coupled with the need to spend more on acquiring customers to make up for those that have churned.

3. Studying this rate also involves keeping tabs on the competition, another critical area of maintaining business success. This is because the CCR can increase when competitors launch new and/or less expensive products, which may entice customers to churn and switch to them. This dampens customer loyalty for obvious reasons and for some customers, can be a permanent switch to another brand. The ability to predict the churn rate is necessary for a company’s long-term success. Thus, businesses should gauge their churn rate and control it whenever possible.

### Objective
The goal of this notebook is to understand and predict customer churn for an Ecommerce store. Specifically, we will initially perform Exploratory Data Analysis (EDA) to identify and visualise the factors contributing to customer churn. This analysis will later help us build Machine Learning models to predict whether a customer will churn or not.

## Data analysis
###  Libraries And Tools
```python
import numpy as np #library for  linear algebra
import pandas as pd # library for creating data frames and  processing data
from itertools import chain # A tool for nested lists
# libraries for plotting the data
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.io as pio
import matplotlib as mpl
import missingno as msno
pyo.init_notebook_mode()
sns.set_style('darkgrid')
plt.rc('figure',figsize=(18,9))
%matplotlib inline
```

### Defining the functions
```python
# this function is used for plotting continuous features

def continuous_plot(feature):
    plt.figure(figsize = (15,4), facecolor = 'white')
    df_retained = data[data['Churn'] == 0]
    df_churned = data[data['Churn'] == 1]
    sns.histplot(data=df_retained, x=feature, bins=20, color='#00A5E0', alpha=0.66, 	 
	  edgecolor='firebrick', kde=True, label = 'Retained')
    sns.histplot(data=df_churned, x=feature, bins=20, color='#DD403A', alpha=0.66,
	edgecolor='firebrick', kde=True, label = 'Churned')
    plt.xlabel = '{}'.format(feature)
    COLOR = 'black'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR
    plt.rcParams['axes.facecolor']='white'
    plt.legend()
    plt.show()
```
```python
# this function is used for plotting categorical features
def categorical_plot(feature):    
    plt.figure(figsize = (15,4), facecolor = 'white')
    df_retained = data[data['Churn'] == 0]
    df_churned = data[data['Churn'] == 1]
    sns.countplot(data=df_retained, x=feature, color='#00A5E0', alpha=0.66,
    edgecolor='firebrick', label = 'Retained')
    sns.countplot(data=df_churned, x=feature, color='#DD403A', alpha=0.66,
	edgecolor='firebrick', label = 'Churned')
    plt.xlabel = '{}'.format(feature)
    COLOR = 'black'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR
    plt.rcParams['axes.facecolor']='white'
    plt.legend()
    plt.show()
```

### Exploratory Data Analysis

```python
data = pd.read_excel('../data/E Commerce Dataset.xlsx', sheet_name='E Comm')
```
```python
data.info()
```
![Screenshot 2021-11-01 at 6 35 22 PM](https://user-images.githubusercontent.com/83364833/139711801-d791809b-886a-42fc-886a-457261588d54.png)

![Screenshot 2021-11-01 at 6 35 37 PM](https://user-images.githubusercontent.com/83364833/139711910-e3e5f8b7-2882-489c-98e4-81b282e00902.png)

### Checking for imbalances in the dataset
```python
#These are the pie figures
colors = ['#87cefa','#f08080']
fig = make_subplots(
    rows=2, cols=2, subplot_titles=('','<b>churn percentage per gender','<b>churn percentage <b>','Residuals'),
    vertical_spacing=0.09,
    specs=[[{"type": "pie","rowspan": 2}       ,{"type": "pie"}] ,
           [None                               ,{"type": "pie"}]            ,                                      
          ]
)

fig.add_trace(
    go.Pie(values = data.Gender.value_counts().values,labels=['<b>male<b>','<b>Female<b>'],hole=0.3,pull=[0,0.3]),
    row=1, col=1
)

fig.add_trace(
    go.Pie(
        labels=['Female customer churn','Male customer churn'],
        values=list(chain.from_iterable(np.array(data[['Churn','Gender']].groupby('Gender').sum()))),
        pull=[0,0.01,0.5],
        hole=0.3
        
    ),
    row=1, col=2
)

fig.add_trace(
    go.Pie(
        labels=['Retained customers','Churned customers'],
        values= list(chain.from_iterable(np.array(data[['Churn','CustomerID']].groupby('Churn').count()))),
        pull=[0,0.2,0.5],
        hole=0.3,
    ),
    row=2, col=2
)
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15,
                  marker=dict(colors=colors, line=dict(color='white', width=10)))


fig.update_layout(
    height=800,
    showlegend=False,
    title_text="<b>Distribution Of Gender  and its churn percentage<b>",
    paper_bgcolor = 'rgb(255,255,255)'
)

fig.show()
```
![newplot (1)](https://user-images.githubusercontent.com/83364833/139712111-63e44dde-2833-4087-96d5-85aabd00bf45.png)

The ecommerce business retained the 83.2% of their customers. Therefore we conclude that this dataset is imbalanced due to the population difference between the retained and churned class, thus, the accuracy metric its not suitable for measuring model performance.

Now, i will group the different types of variables  into two categories. the categorical and the continuous, in order to apply different visualisation 
techniques. I will exclude the columns of  `CustomerID`  because it has no value in the analysis.


![Screenshot 2021-11-01 at 6 44 24 PM](https://user-images.githubusercontent.com/83364833/139712356-7ecde2b9-681c-48c4-a315-96567064642c.png)

### Analysis of Continuous Variables
Firstly, i will compute the standard correlation coefficient between every pair of continuous variables using heatmap().

```python
data_cont = data[continuous[:continuous.index("-")]]

# This is the heatmap

fig1 = go.Figure(data=go.Heatmap(
        z = data_cont.corr(),
        x = continuous,
        y = continuous,
        colorscale = 'blues'))

fig1.update_layout(
    height=600,
    width = 1000,
    showlegend=True,
    title_text="<b>Correlation Matrix of continuous variables<b>",
    paper_bgcolor = 'rgb(255,255,255)'
)

fig1.show()
```
![newplot (1) copy](https://user-images.githubusercontent.com/83364833/139712533-5b7925db-3279-4904-b275-9f210ce3eb96.png)

Then we will plot each variable indepedently and display a sample of these plots below

#### Number Of Address analysis
![Screenshot 2021-11-01 at 6 46 28 PM](https://user-images.githubusercontent.com/83364833/139713006-59743004-2a8a-4249-a855-289bc530e51c.png)

#### DaySinceLastOrder analysis

![Screenshot 2021-11-01 at 6 46 55 PM](https://user-images.githubusercontent.com/83364833/139713115-754d2424-20ea-4084-96ef-e15ca76f6e02.png)

#### CashbackAmount analysis
![Screenshot 2021-11-01 at 6 47 15 PM](https://user-images.githubusercontent.com/83364833/139713218-1d6eb530-5439-4f07-9e20-30f16d659a01.png)

### Keypoints of numerical variables
1. There is no significant difference between retained and churned customers in terms of the feature WarehouseToHome .
2. There is no significant difference between retained and churned customers in terms of the feature NumberOfAddress .
3. There is no significant difference between retained and churned customers in terms of the feature OrderAmountHikeFromlastYear
4. There is no significant difference between retained and churned customers in terms of the feature CouponUsed
5. There is no significant difference between retained and churned customers in terms of the feature OrderCount
6. There is no significant difference between retained and churned customers in terms of the feature DaySinceLastOrder 
7. There is no significant difference between retained and churned customers in terms of the feature CashbackAmount

### Analysis of Categorical Variables
below are some samples of the plots

![Screenshot 2021-11-01 at 6 47 24 PM](https://user-images.githubusercontent.com/83364833/139714102-c0a7721b-2199-4919-997a-b6d87d67c3a5.png)

![Screenshot 2021-11-01 at 6 47 48 PM](https://user-images.githubusercontent.com/83364833/139714152-142817be-c35f-4fab-b461-ad98aaff4f88.png)

![Screenshot 2021-11-01 at 6 48 07 PM](https://user-images.githubusercontent.com/83364833/139714187-b0454912-e475-4800-a2a8-f3e4b1b58271.png)


### Keypoints of categorical variables
1. We can conclude that the vast majority of churned customers had a subscription of one month or less .
2. It's not a surprise that most churned customers spent less than four hours on app.
3. It seems that different devices does not affect the churned rate.
4. Most churn customers had been registered in more than three devices.
5. It seems that different complain status does not affect the churned rate.
6. The majority of churned customers use Credit and Debit card.
7. The vast majority of churned customers prefered categories of technology such as mobiles and laptops
8. Suprisingly, most of churned customers gave a satisfaction score of more than three
9. It seems that single customers had the biggest churn rate

## Cleaning the data
### Libraries & Tools
```python
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder,OneHotEncoder ,MinMaxScaler
```
### Feature Selection
In this section we will drop all features tha does not provide any value in our target variable. At first we will drop the CustomerID as it is irrelevant to the target value. Secondly we will perform chi-square test on categorical features in order to see, which of them may confirm the initial hypothesis of being redundant.

```python
categorical = ['Tenure', 'HourSpendOnApp','PreferredLoginDevice', 'CityTier', 'NumberOfDeviceRegistered', 'Complain', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus']
```
```python
chi2_array, p_array = [], []
for column in categorical:

    crosstab = pd.crosstab(data[column],data['Churn'])
    # Defining cross tabulation 
    '''Cross Tabulation. also known as contingency tables or cross tabs,
    cross tabulation groups variables to understand the correlation between different variables. 
    It also shows how correlations change from one variable grouping to another.
    It is usually used in statistical analysis to find patterns, trends,
    and probabilities within raw data.'''
    
    chi2, p, dof, expected = chi2_contingency(crosstab)
    chi2_array.append(chi2)
    p_array.append(p)

df_chi = pd.DataFrame({
    'Variable': categorical,
    'Chi-square': chi2_array,
    'p-value': p_array
})
df_chi.sort_values(by='Chi-square', ascending=False)
```
![Screenshot 2021-11-03 at 9 18 04 AM](https://user-images.githubusercontent.com/83364833/140021630-57553a46-d604-46f1-a826-316d76400ee5.png)

According to the table, the feature HourSpendOnApp has a small chi-square a p-value greater than 0.05 which is the standard cut-off value .Therefore our initial hypothesis is confirmed and HourSpendOnApp does not convey any useful information. In the next step, i will drop all the unnecesary columns and null values of the dataset

```python
data_model = data.dropna() # dropping null values

data_model = data_model.drop(columns = ['CustomerID','HourSpendOnApp' ]) # dropping unnecesary columns
```
### Encoding Categorical Features
In order to implement machine learning algorithms, we have to convert(encode) all categorical features to numbers.

On our dataset, five categorical features require encoding.

1. for ```PreferredLoginDevice``` , ```Gender``` , we will use scikit-learn's ```LabelEncoder()``` which maps each unique label to an integer


2. for ```PreferredPaymentMode```  i will map the values as:
Debit card = 1,
E-wallet = 2,
credit card = 3,
Other methods = 4
in order to to make all values equally important to the feature

3. for ```PreferedOrderCat``` i will map the values as:
Laptop & Accessory = 1,
Mobile = 2,
Mobile Phone = 3,
Other = 4
in order to to make all values equally important to the feature

```python
data_model['PreferredLoginDevice'] =
LabelEncoder().fit_transform(data_model['PreferredLoginDevice'])

data_model['Gender'] =
LabelEncoder().fit_transform(data_model['Gender'])

data_model['MaritalStatus'] =
LabelEncoder().fit_transform(data_model['MaritalStatus'])

data_model['PreferredPaymentMode'] = data_model['PreferredPaymentMode'].map({'Debit Card': 1,'E-wallet': 2,'Credit Card': 3, 'CC':4, 'COD':4, 'UPI':4, 'Cash on Delivery':4 })

data_model['PreferedOrderCat'] = data_model['PreferedOrderCat'].map({'Laptop & Accessory': 1,'Mobile': 2,'Mobile Phone': 3, 'Others':4, 'Fashion':4, 'Grocery':4 })
```
#### Exporting the model_data
```python
model_data.to_csv('model_data')
```
## Modeling
### Libraries & Tools

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest,mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from pactools.grid_search import GridSearchCVProgressBar
```
### Dealing with the imabalance
As we saw in the analysis notebook, the dataset is imbalanced against the churned class, therefore we will implement an oversampling technique called Adasyn. The major difference between SMOTE (the most widely used technique) and ADASYN is the difference in the generation of synthetic sample points for minority data points. In ADASYN, we consider a density distribution rₓ which thereby decides the number of synthetic samples to be generated for a particular point, whereas in SMOTE, there is a uniform weight for all minority points.

```python
X = data_model.drop('Churn', axis = 1)
y = data_model['Churn']

ada = ADASYN(sampling_strategy='minority', n_neighbors=5, n_jobs=-1)

X_res, y_res = ada.fit_resample(X,y)

y_res = pd.DataFrame(y_res, columns = ['Churn'])

X_train, X_test, y_train, y_test = train_test_split(X_tr_df, y_res, test_size=0.33, random_state=42)
```
### Creating our models
At first will create six different models with different classifiers and i will implement CV grid search in each one of them. At last, i will evaluate them using precission, accuracy and auc scores.
```python
scoring = pd.DataFrame(columns = ['classifier','precission','accuracy','roc_auc_score'])
```
### SVC
```python
param_svc = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

cv = StratifiedKFold()
#Metrics for Evualation:
met_grid= 'accuracy' #The metric codes from sklearn
svc = RandomizedSearchCV(SVC(), param_svc, scoring=met_grid, refit='roc_auc_score', return_train_score=True, verbose = True)
svc.fit(X_train,  y_train.values.ravel())
svc_preds = svc.predict(X_test)

scoring = scoring.append({'classifier':'SVC',
                          'precission':precision_score(y_test,svc_preds),
                          'accuracy':accuracy_score(y_test,svc_preds),
                          'roc_auc_score':roc_auc_score(y_test,svc_preds)}, 
						  ignore_index= True)
```
### Random Forest Classifier
```python
param_rfc = {
             'max_depth': [10,50 , 100, None],
             'min_samples_leaf': [1, 2, 4],
             'n_estimators': [200,800,1600]}

cv = StratifiedKFold()
#Metrics for Evualation:
met_grid= 'accuracy' #The metric codes from sklearn
rfc = RandomizedSearchCV(RandomForestClassifier(), param_rfc, scoring= met_grid, refit='roc_auc_score', return_train_score=True, verbose = True, n_jobs=-1)
rfc.fit(X_train,  y_train.values.ravel())
rfc_preds = rfc.predict(X_test)

scoring = scoring.append({'classifier':'Random Forest',
                          'precission':precision_score(y_test,rfc_preds),
                          'accuracy':accuracy_score(y_test,rfc_preds),
                          'roc_auc_score':roc_auc_score(y_test,rfc_preds)}, 
						  ignore_index= True)
```
### ADA Boost
```python
param_ada = {
             'learning_rate': [ 0.01, 0.1, 1, 10],
             'n_estimators': [200,800,1600]}

cv = StratifiedKFold()
#Metrics for Evualation:
met_grid= 'accuracy' #The metric codes from sklearn
adaboost = RandomizedSearchCV(AdaBoostClassifier(), param_ada, scoring= met_grid, refit='roc_auc_score', return_train_score=True, verbose = True, n_jobs=-1)
adaboost.fit(X_train,  y_train.values.ravel())
adaboost_preds = adaboost.predict(X_test)

scoring = scoring.append({'classifier':'Ada boost',
                          'precission':precision_score(y_test,adaboost_preds),
                          'accuracy':accuracy_score(y_test,adaboost_preds),
                          'roc_auc_score':roc_auc_score(y_test,adaboost_preds)}, 
						  ignore_index= True)
```
### Gradient Boosting
```python
param_gdc = {'n_estimators': [200,800,1600],
             'learning_rate': [ 0.1, 1,10]
             }

cv = StratifiedKFold()
#Metrics for Evualation:
met_grid= 'accuracy' #The metric codes from sklearn
gdc = RandomizedSearchCV(GradientBoostingClassifier(), param_gdc, scoring= met_grid, refit='roc_auc_score', return_train_score=True, verbose = True, n_jobs=-1)
gdc.fit(X_train,  y_train.values.ravel())
gdc_preds = gdc.predict(X_test)

scoring = scoring.append({'classifier':'Gradient boost',
                          'precission':precision_score(y_test,gdc_preds),
                          'accuracy':accuracy_score(y_test,gdc_preds),
                          'roc_auc_score':roc_auc_score(y_test,gdc_preds)}, 
						  ignore_index= True)
```

### XGBoost
```python
param_xgb = {'gamma': [0.5,1.5, 2, 5],
             'learning_rate': [ 0.1, 1,10],
             'max_depth': [3, 4, 5]}

cv = StratifiedKFold()
#Metrics for Evualation:
met_grid= 'accuracy' #The metric codes from sklearn
xgb = RandomizedSearchCV(XGBClassifier(use_label_encoder=False), param_xgb, scoring= met_grid, refit='roc_auc_score', return_train_score=True, verbose = True, n_jobs=-1)
xgb.fit(X_train,  y_train.values.ravel())
xgb_preds = xgb.predict(X_test)

scoring = scoring.append({'classifier':'XGBoost',
                          'precission':precision_score(y_test,xgb_preds),
                          'accuracy':accuracy_score(y_test,xgb_preds),
                          'roc_auc_score':roc_auc_score(y_test,xgb_preds)}, ignore_index= True)
```

### Logistic Regression
```
param_log = [{'penalty': ['l1', 'l2'],
              'C': [1,5],
              'solver': ['liblinear']}] 

cv = StratifiedKFold()
#Metrics for Evualation:
met_grid= 'accuracy' #The metric codes from sklearn
log = RandomizedSearchCV(LogisticRegression(), param_log, scoring= met_grid, refit='roc_auc_score', return_train_score=True, verbose = True, n_jobs=-1)
log.fit(X_train,  y_train.values.ravel())
log_preds = log.predict(X_test)

scoring = scoring.append({'classifier':'Logistic Regression',
                          'precission':precision_score(y_test,log_preds),
                          'accuracy':accuracy_score(y_test,log_preds),
                          'roc_auc_score':roc_auc_score(y_test,log_preds)}, 
						  ignore_index= True)
```

### Plotting the results

```python
scoring = scoring.sort_values(by = ['precission','accuracy','roc_auc_score'], ascending = False)
```
```python
classifiers=scoring['classifier'].values.tolist()
fig=go.Figure()
for clf,color in zip(classifiers, px.colors.sequential.Blues[::-1]):
    fig.add_trace(go.Bar(name=clf, x=scoring.drop('classifier', axis = 1).keys().tolist(),
                                   y=scoring.drop('classifier', axis = 1)[scoring['classifier']==clf].values.tolist()[0],
                                   marker_color = color,
                                   marker_line_color='rgb(8,48,107)',
                                   marker_line_width=1.5,
                                   opacity=0.6
                                   ))
# Change the bar mode
fig.update_layout( title=" classifier score per metric",
                    xaxis_title="metric",
                    yaxis_title="value",
                    plot_bgcolor='white',
                    )
fig.show()
```
As we can conclude, Gradient boost algorithm scored the best in all of these metrics, therefore we will continue our modeling with this algorithm

![Screenshot 2021-11-03 at 9 39 41 AM](https://user-images.githubusercontent.com/83364833/140023834-cd4facca-2357-4467-ac29-ef8ecd77c00e.png)
### Fine tuning the final model

In the final tuning of the alogortithm i will run an exhaustive grid search in a pipeline that it will contain thealgorithm and a feature selection algorithm called SelectKbest. Feature selection is a technique where we choose those features in our data that contribute most to the target variable. In other words we choose the best predictors for the target variable.

the main advantages are:
1. Reduces Overfitting: Less redundant data means less possibility of making decisions based on redundant data/noise.
2. Improves Accuracy: Less misleading data means modeling accuracy improves.
3. Reduces Training Time: Less data means that algorithms train faster.

```python
param_pipe = {'feature_selection__k': [5,10,20],
             'classifier__n_estimators': [1000,1400,1600,2000],
             'classifier__learning_rate': [ 0.1, 1,10],
             'classifier__max_depth': [3,10,20],
             }

pipe = Pipeline([
                 ('feature_selection',SelectKBest(score_func=mutual_info_classif)),
                 ('classifier',GradientBoostingClassifier())
                ])


#GridSearchCVProgressBar is identical to GridSearchCV, but it adds a nice progress bar to monitor progress (pactools library).
pipe = GridSearchCVProgressBar(pipe, param_pipe, scoring= 'accuracy', refit='roc_auc_score', return_train_score=True,cv=3 ,verbose = 1 , n_jobs=-1)
pipe.fit(X_train,  y_train.values.ravel())
pipe_preds = pipe.predict(X_test)
```
```python
pipe.best_params_
```


![Screenshot 2021-11-03 at 9 43 31 AM](https://user-images.githubusercontent.com/83364833/140024217-471e8e4a-0187-479b-9cf1-b71c0e7b7900.png)

**please refer to the dashboard for the explanattion of the model**
