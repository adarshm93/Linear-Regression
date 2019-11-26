# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:48:25 2019

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

# loading the data
startup=pd.read_csv("E:/ADM/Excelr solutions/DS assignments/multi linear regression/50_Startups.csv")

# to get top rows
startup.head(40) 

#Create dummies for categorical data
dummy=pd.get_dummies(startup["State"])
startup2 = pd.concat([startup, dummy], axis=1)
#drop categorical variable after creating dummies
startup2 = startup2.drop("State", axis=1)

#renaming column names for convineance
startup2 = startup2.rename({'R&D Spend': 'RSpend', 'Marketing Spend': 'Marketing_Spend','New York': 'New_York'}, axis=1)
startup2.columns
# Correlation matrix 
a=startup2.corr()

#EDA
a1=startup2.describe()
startup2.median()
startup2.var()
startup2.skew()
plt.hist(startup2["R&D Spend"])
plt.hist(startup2["Administration"])
plt.hist(startup2["Marketing Spend"])
plt.hist(startup2["Profit"])

plt.boxplot(startup2["R&D Spend"],0,"rs",0) #No  outliers present
plt.boxplot(startup2["Administration"],0,"rs",0)
plt.boxplot(startup2["Marketing Spend"],0,"rs",0)
plt.boxplot(startup2["Profit"],0,"rs",0)

startup2.isnull().sum() # no null values present


#sctterplot and histogram between variables
sns.pairplot(startup2)#RD_spend and Marketspend is in colinearity

### Splitting the data into train and test data 
startup_train,startup_test  = train_test_split(startup2,test_size = 0.3) # 30% size

# preparing the model on train data 
model_train = smf.ols("Profit~RSpend+Administration+Marketing_Spend+California+Florida+New_York",data=startup_train).fit()
model_train.summary()# Administration and Marketing_spend is not significant

#preparing model based on only Administration
ml_a=smf.ols("Profit~Administration",data=startup_train).fit()
ml_a.summary()#insignificant

ml_m=smf.ols("Profit~Marketing_Spend",data=startup_train).fit()
ml_m.summary()#significant

ml_am=smf.ols("Profit~Administration+Marketing_Spend",data=startup_train).fit()
ml_am.summary()#both significant

# calculating VIF's values of independent variables
rsq_RSpend = smf.ols('RSpend~Administration+Marketing_Spend+California+Florida+New_York',data=startup_train).fit().rsquared  
vif_RSpend = 1/(1-rsq_RSpend) # 2.42

rsq_ad = smf.ols('Administration~RSpend+Marketing_Spend+California+Florida+New_York',data=startup_train).fit().rsquared  
vif_ad = 1/(1-rsq_ad) # 1.26

rsq_ms = smf.ols('Marketing_Spend~RSpend+Administration+California+Florida+New_York',data=startup_train).fit().rsquared  
vif_ms = 1/(1-rsq_ms) #  2.13

   # Storing vif values in a data frame
d1 = {'Variables':['RSpend','Administration','Marketing_Spend'],'VIF':[vif_RSpend,vif_ad,vif_ms]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(model_train)

# Checking whether data has any influential values 
# influence index plots
sm.graphics.influence_plot(model_train)
#drop
startup2_new=startup_train.drop(startup_train.index[[17]],axis=0)


# preparing the new model on train data 
model_train_new = smf.ols("Profit~RSpend+Administration+Marketing_Spend+California+Florida+New_York",data=startup2_new).fit()
model_train_new.summary()# Administration and Marketing_spend is not significant
# influence index plots
sm.graphics.influence_plot(model_train_new)
#drop
startup2_new2=startup2_new.drop(startup2_new.index[[2]],axis=0)

#new model for new dataset
model_train_new2 = smf.ols("Profit~RSpend+Administration+Marketing_Spend+California+Florida+New_York",data=startup2_new2).fit()
model_train_new2.summary()#
sm.graphics.influence_plot(model_train_new2)
startup2_new3=startup2_new2.drop(startup2_new2.index[[7]],axis=0)

#new model for new dataset
model_train_new3 = smf.ols("Profit~RSpend+Administration+Marketing_Spend+California+Florida+New_York",data=startup2_new3).fit()
model_train_new3.summary()
sm.graphics.influence_plot(model_train_new3)
startup2_new4=startup2_new3.drop(startup2_new3.index[[7]],axis=0)

#new model
model_train_new4 = smf.ols("Profit~RSpend+Administration+Marketing_Spend+California+Florida+New_York",data=startup2_new4).fit()
model_train_new4.summary()
sm.graphics.influence_plot(model_train_new4)
startup2_new5=startup2_new4.drop(startup2_new4.index[[18,19]],axis=0)


model_train_new5 = smf.ols("Profit~RSpend+Administration+Marketing_Spend+California+Florida+New_York",data=startup2_new5).fit()
model_train_new5.summary()#marketing spend is again insignificant so build new model without this varibale.


####################################################################################
#markete spend showing insignificant, so dont consider this column
model=smf.ols("Profit~RSpend+Administration+California+Florida+New_York",data=startup2_new5).fit()#without marketing spend
model.summary()#0.974
####################################################################################


# Predicted values of Profit
profit_pred = model.predict(startup2_new5)
profit_pred

resid=profit_pred-startup2_new5.Profit #av-fitted values
np.sum(resid)#sum of residuals nearly zero


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(startup2_new5.Profit,profit_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(profit_pred,model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

########    Normality plot for residuals ######
# histogram
plt.hist(model.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

st.probplot(model.resid_pearson, dist="norm", plot=pylab)

############ Homoscedasticity #######
# Residuals VS Fitted Values 
plt.scatter(profit_pred,model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# train_data prediction
train_pred = model.predict(startup2_new5)

# train residual values 
train_resid  = train_pred - startup2_new5.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))#10937.052

# prediction on test data set 
test_pred = model.predict(startup_test)
# test residual values 
test_resid  = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))#11012.209

#Accurecy
train_pred.corr(startup2_new5.Profit)#0.9679
test_pred.corr(startup_test.Profit)#0.9840

################################################

#Quadratic transformation
model2=smf.ols("Profit~(RSpend+RSpend*RSpend)+(Administration+Administration*Administration)+(California+California*California)+(Florida+Florida*Florida)+(New_York+New_York*New_York)",data=startup2_new5).fit()
model2.summary()#insignificant

# train_data prediction
train_pred2 = model2.predict(startup2_new5)
# train residual values 
train_resid2  = train_pred2 - startup2_new5.Profit
# RMSE value for train data 
train_rmse2 = np.sqrt(np.mean(train_resid2*train_resid2))#9391.2245

# prediction on test data set 
test_pred2 = model2.predict(startup_test)
# test residual values 
test_resid2  = test_pred2 - startup_test.Profit
# RMSE value for test data 
test_rmse2 = np.sqrt(np.mean(test_resid2*test_resid2))#9004.75

##########################################################

#Square Transformation
model3=smf.ols("Profit~(RSpend*RSpend)+(Administration*Administration)+(California*California)+(Florida*Florida)+(New_York*New_York)",data=startup2_new5).fit()
model3.summary()

# train_data prediction
train_pred3 = model3.predict(startup2_new5)
# train residual values 
train_resid3  = train_pred3 - startup2_new5.Profit
# RMSE value for train data 
train_rmse3 = np.sqrt(np.mean(train_resid3*train_resid3))#10205.2346

# prediction on test data set 
test_pred3 = model3.predict(startup_test)
# test residual values 
test_resid3  = test_pred3 - startup_test.Profit
# RMSE value for test data 
test_rmse3 = np.sqrt(np.mean(test_resid3*test_resid3))#7642.2513


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#Train RMSE and Test RMSE is nearly equal for simple model so its good model with Accurecy =96% , model 2 is overfit model.
 
#Parameters	     simple final model         quadratic transformation     square transformation
   
   R-squr value	     0.974                         0.974                          0.952
           Rmse      train and test same           overfit                       good model
	    P-value	     All are significant         market spend>0.05              market spend >0.05


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''