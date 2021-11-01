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
