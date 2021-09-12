# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:08:11 2020

@author: A2024
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor


cleaned_data = pd.read_csv('C:\\Users\\A2024\\Downloads\\Cleaned_Data.csv')
cleaned_data.info()
descr_hp=cleaned_data.describe()
cleaned_data.columns
cleaned_data.shape
cleaned_data.dtypes.value_counts()


hp_data=cleaned_data.drop(['PropType','Taxkey','Address','CondoProject'],axis=1)
hp_data['year_sale'] = hp_data['Sale_date'].str[:4]
hp_data['house_age']=hp_data['year_sale'].astype(int)-hp_data['Year_Built']
hp_data=hp_data.loc[(hp_data['house_age']>=0)]

hp_data['Tbath']=hp_data['Fbath']+0.5*hp_data['Hbath']
hp_data.columns

hp_data_fin=hp_data.drop(['Year_Built','Hbath','Fbath','Sale_date'],axis=1)
is_null_counts = hp_data_fin.isnull().sum().sort_values(ascending = False)
hp_data_fin['Extwall'].unique()


numerical_cols = hp_data_fin.dtypes[hp_data_fin.dtypes != 'object'].index
numerical_data = hp_data_fin[numerical_cols]

num_corr = numerical_data.corr()['Sale_price'].abs().sort_values(ascending = False)
num_corr

corr_matrix = numerical_data.corr().sort_values('Sale_price')
corr_target = abs(corr_matrix)
sns.heatmap(corr_matrix, 
            xticklabels=corr_matrix.columns.values,
            yticklabels=corr_matrix.columns.values,
            annot = True, cmap="YlGnBu")
#

fin_descr=hp_data_fin.describe()


######CATEGORICAL
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'District', y = 'Sale_price', data = hp_data_fin)
plt.subplot(2,3,2)
sns.boxplot(x = 'Style', y = 'Sale_price', data = hp_data_fin)
plt.subplot(2,3,3)
sns.boxplot(x = 'Extwall', y = 'Sale_price', data = hp_data_fin)
plt.subplot(2,3,4)
sns.boxplot(x = 'Stories', y = 'Sale_price', data = hp_data_fin)
plt.subplot(2,3,5)
sns.boxplot(x = 'year_sale', y = 'Sale_price', data = hp_data_fin)
plt.subplot(2,3,6)
sns.boxplot(x = 'Bdrms', y = 'Sale_price', data = hp_data_fin)
plt.show()



###########DATA CREATION############
#dummy

district = pd.get_dummies(hp_data_fin['District'])

style = pd.get_dummies(hp_data_fin['Style'])
Extwall = pd.get_dummies(hp_data_fin['Extwall'])


#########Filter data based on year_sale
plt.figure(figsize = (20,10))
ax = sns.barplot(x = 'year_sale', y = 'Sale_price', data = hp_data_fin, palette = ['gray', 'gray', 'steelblue', 'lightsteelblue', 'lightsteelblue', 'steelblue', 'gray', 'gray', 'gray'], linewidth = 1.5, edgecolor = 'black')
plt.show()


plt.scatter(hp_data_fin['year_sale'],hp_data_fin['Sale_price'],label='True Position')
plt.xlabel("year_sale")
plt.ylabel("Sale_price")
plt.show()
#no trend

hp_data_fin['year_sale']=hp_data_fin['year_sale'].astype(int)
des=hp_data_fin.describe()

plt.hist(hp_data_fin['year_sale'])
###########CONCAT DUMMIES#########3


hp_data_fin_1 = pd.concat([hp_data_fin, district,style,Extwall], axis = 1)

####################REMOVE YEAR_SALE
hp_data_fin_1.columns

hp_data_fin_1=hp_data_fin_1.drop(['District','Nbhd','Style','Extwall'],axis=1)
hp_data_fin_1=hp_data_fin_1.loc[(hp_data_fin_1['year_sale']>=2009)]
hp_data_fin_1=hp_data_fin_1.drop(['year_sale'],axis=1)

#hp_data_fin_1=hp_data_fin_1.loc[(hp_data_fin_1['Sale_price']>=5000)]
#hp_data_fin_1=hp_data_fin_1.loc[(hp_data_fin_1['Sale_price']<1650000)]

##########SCALING############  #####should scaling be on whole data?#####
scaler = StandardScaler()
numerical_vars=['Stories', 'Fin_sqft', 'Units',
                     'Bdrms',  'Lotsize',
                 'house_age', 'Tbath']

#hp_data_fin_1[numerical_vars] = scaler.fit_transform(hp_data_fin_1[numerical_vars])



########SPLIT DATA#####3

np.random.seed(0)
df_train, df_test = train_test_split(hp_data_fin_1, train_size = 0.8, test_size = 0.2, random_state = 100)



########### X an Y
y_train = df_train.pop('Sale_price')
X_train = df_train

X_train[numerical_vars] = scaler.fit_transform(X_train[numerical_vars])


#########FIT

lm = LinearRegression()
lm.fit(X_train, y_train)


#######recursive feature elimination #####READ
rfe = RFE(lm,22)             # running RFE
rfe = rfe.fit(X_train, y_train)
#list(zip(X_train.columns,rfe.support_,rfe.ranking_))


col = X_train.columns[rfe.support_]
col

X_train.columns[~rfe.support_]

# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]



# Calculate the VIFs for the model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)

lm = sm.OLS(y_train,X_train_rfe).fit()  
print(lm.summary())

y_train_pred = lm.predict(X_train_rfe)

mean_squared_error(y_train,y_train_pred,squared=False )

load_train=y_train.describe()




####RESIDUALS PLOT
y_train_price = lm.predict(X_train_rfe)
res = (y_train_price - y_train)

fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)  


#####Test###########

y_test = df_test.pop('Sale_price')
X_test = df_test

X_test[numerical_vars] = scaler.fit_transform(X_test[numerical_vars])


X_test = sm.add_constant(X_test)
X_test_rfe = X_test[X_train_rfe.columns]
y_pred = lm.predict(X_test_rfe)
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)

# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)

######UNSCALE

#y_new_inverse = scaler.inverse_transform(y_test)


######CHECK RMSE 

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,y_pred,squared=False )

load=y_test.describe()


#######RF################

#X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      #random_state=0)
model_5 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

#model_3 = RandomForestRegressor(n_estimators=10, criterion='mae', random_state=0)
X_train_rfe = X_train[col]
X_test_rfe = X_test[col]



model_train_1=model_5.fit(X_train_rfe,y_train)
preds = model_train_1.predict(X_test_rfe)
mean_squared_error(y_test, preds,squared=False )

preds_train = model_train_1.predict(X_train_rfe)
mean_squared_error(y_train, preds_train,squared=False )

r2_f = 1-(sum((y_train-preds_train)**2)/sum((y_train-np.mean(y_train))**2))

columns=X_train_rfe.columns
importances = model_train_1.feature_importances_
importances=pd.DataFrame(importances,index=columns)
importances=importances.sort_values(by=0,ascending=False)
#importances_1=importances.loc[(importances[0]>=0.01)]


#model_train_2=model_3.fit(X_train,y_train)
#preds = model_train_2.predict(X_test)

mean_squared_error(y_test, preds,squared=False )

mae_f = np.mean(abs(y_test-preds))
r2_f = 1-(sum((y_test-preds)**2)/sum((y_test-np.mean(y_test))**2))
mse_f = np.mean((y_test-preds)**2)
rmse_f = np.sqrt(mse_f)

x = list(range(len(y_test)))
plt.scatter(x, y_test, color="blue", label="original")
plt.plot(x, preds, color="red", label="predicted")
plt.legend()
plt.show() 



 
 
 
#################END##################


###################EDA####################


plt.scatter(cleaned_data['Fin_sqft'],cleaned_data['Sale_price'],label='True Position')
#linear

plt.scatter(cleaned_data['Bdrms'],cleaned_data['Sale_price'],label='True Position')
plt.xlabel("Bdrms")
plt.ylabel("Sale_price")
plt.show()
#normally distributed

plt.scatter(cleaned_data['Fbath'],cleaned_data['Sale_price'],label='True Position')
plt.xlabel("Fbath")
plt.ylabel("Sale_price")
plt.show()

plt.scatter(cleaned_data['Hbath'],cleaned_data['Sale_price'],label='True Position')
plt.xlabel("Hbath")
plt.ylabel("Sale_price")
plt.show()



plt.scatter(cleaned_data['Lotsize'],cleaned_data['Sale_price'],label='True Position')
plt.xlabel("Lotsize")
plt.ylabel("Sale_price")
plt.show()
#no trend


plt.scatter(cleaned_data['Units'],cleaned_data['Sale_price'],label='True Position')
plt.xlabel("Units")
plt.ylabel("Sale_price")
plt.show()
#no trend

plt.scatter(cleaned_data['District'],cleaned_data['Sale_price'],label='True Position')
plt.xlabel("District")
plt.ylabel("Sale_price")
plt.show()
#no trend



plt.figure(figsize = (20,10))
ax = sns.barplot(x = 'District', y = 'Sale_price', data = cleaned_data, palette = ['gray', 'gray', 'steelblue', 'lightsteelblue', 'lightsteelblue', 'steelblue', 'gray', 'gray', 'gray'], linewidth = 1.5, edgecolor = 'black')
plt.show()



###OUTLIER

fig, axs = plt.subplots(2,4, figsize = (10,5))
plt1 = sns.boxplot(hp_data_fin['Stories'], ax = axs[0,0])
plt2 = sns.boxplot(hp_data_fin['Fin_sqft'], ax = axs[0,1])
plt3 = sns.boxplot(hp_data_fin['Units'], ax = axs[0,2])
plt4 = sns.boxplot(hp_data_fin['Tbath'], ax = axs[0,3])
plt1 = sns.boxplot(hp_data_fin['Bdrms'], ax = axs[1,0])
plt2 = sns.boxplot(hp_data_fin['Lotsize'], ax = axs[1,1])
plt3 = sns.boxplot(hp_data_fin['house_age'], ax = axs[1,2])
plt4 = sns.boxplot(hp_data_fin['Sale_price'], ax = axs[1,3])
plt.tight_layout()


sns.pairplot(hp_data_fin)
plt.show()
hp_data_fin.columns