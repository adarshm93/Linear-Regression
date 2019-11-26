# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:21:16 2019

@author: Adarsh
"""

# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # for regression model
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import pylab          
import scipy.stats as st

# loading the data
computer=pd.read_csv("E:/ADM/Excelr solutions/DS assignments/multi linear regression/Computer_Data.csv")

# to get top rows
computer.head(40) 
computer.columns
#Create dummies for categorical data
comp=pd.get_dummies(computer)
comp=comp.drop("Unnamed: 0",axis=1)
comp.columns
# Correlation matrix 
a=comp.corr()

#EDA
a1=comp.describe()
comp.median()
comp.var()
comp.skew()
plt.hist(comp["price"])
plt.hist(comp["speed"])
plt.hist(comp["hd"])
plt.hist(comp["ram"])
plt.hist(comp["screen"])

plt.boxplot(comp["price"],0,"rs",0) #No  outliers present
plt.boxplot(comp["speed"],0,"rs",0)
plt.boxplot(comp["hd"],0,"rs",0)
plt.boxplot(comp["ram"],0,"rs",0)
plt.boxplot(comp["screen"],0,"rs",0)

comp.isnull().sum() # no null values present

#sctterplot and histogram between variables
sns.pairplot(comp)#hd-ram in colinearity

### Splitting the data into train and test data 
comp_train,comp_test  = train_test_split(comp,test_size = 0.3) # 30% size
comp_train = comp_train.reset_index()
comp_train=comp_train.drop("index",axis=1)

# preparing the model on train data 
model_train = smf.ols("price~speed+hd+ram+screen+ads+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes+trend",data=comp_train).fit()
model_train.summary()# cd_no is insignificant

#preparing model based on cd_no
ml_cd=smf.ols("price~cd_no",data=comp_train).fit()
ml_cd.summary()#significant

# influence index plots
sm.graphics.influence_plot(model_train)
#drop
comp2=comp_train.drop(comp_train.index[[1281,4279,4116]],axis=0)

#build model after removing influenced observations
model1 = smf.ols("price~speed+hd+ram+screen+ads+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes+trend",data=comp2).fit()
model1.summary()#cd_no insignificant
# influence index plots
sm.graphics.influence_plot(model1)
#drop
comp3=comp2.drop(comp2.index[[1767,2509,2880]],axis=0)
#build model after removing influenced observations
model2 = smf.ols("price~speed+hd+ram+screen+ads+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes+trend",data=comp3).fit()
model2.summary()#insignificant cd_no

#after removing many influenced observations cd_no is insignificant so neglect and dont take in model.
model_f = smf.ols("price~speed+hd+ram+screen+ads+cd_yes+multi_no+multi_yes+premium_no+premium_yes+trend",data=comp3).fit()
model_f.summary()#multi_no is insignificant, remove

#############final model###############
ml_final=smf.ols("price~speed+hd+ram+screen+ads+cd_yes+multi_yes+premium_no+premium_yes+trend",data=comp3).fit()
ml_final.summary()#all variables are significant

# calculating VIF's values of independent variables
rsq_hd = smf.ols('hd~speed+ram+screen+ads+cd_yes+multi_yes+premium_no+premium_yes+trend',data=comp3).fit().rsquared  
vif_hd = 1/(1-rsq_hd) # 4.17

rsq_ram = smf.ols('ram~speed+hd+screen+ads+cd_yes+multi_yes+premium_no+premium_yes+trend',data=comp3).fit().rsquared  
vif_ram = 1/(1-rsq_ram) # 2.95

# Added varible plot 
sm.graphics.plot_partregress_grid(ml_final)

# Predicted values of Profit
price_pred = ml_final.predict(comp3)
price_pred

resid=price_pred-comp3.price #av-fitted values
np.sum(resid)#sum of residuals nearly zero

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(comp3.price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(price_pred,ml_final.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

########    Normality plot for residuals ######
# histogram
plt.hist(ml_final.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
st.probplot(ml_final.resid_pearson, dist="norm", plot=pylab)#normal

############ Homoscedasticity #######
# Residuals VS Fitted Values 
plt.scatter(price_pred,ml_final.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# train_data prediction
train_pred = ml_final.predict(comp3)

# train residual values 
train_resid  = train_pred - comp3.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))#272.371

# prediction on test data set 
test_pred = ml_final.predict(comp_test)
# test residual values 
test_resid  = test_pred - comp_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))#277.52
#Accurecy
train_pred.corr(comp3.price)#0.8841
test_pred.corr(comp_test.price)#0.8741

'''
#Training RMSE and Testing RMSE is nearly same so its good fit model, accurecy=88%

'''
