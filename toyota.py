# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:50:57 2019

@author: Adarsh
"""

#multiple linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf 
import seaborn as sns
from sklearn.model_selection import train_test_split
import pylab
import scipy.stats as st
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#loading the data
corolla=pd.read_csv("E:/ADM/Excelr solutions/DS assignments/multi linear regression/ToyotaCorolla.csv",encoding='ISO-8859-1')
corolla=corolla[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

# to get top rows
corolla.head(40) 
corolla.columns

#EDA
a1=corolla.describe()
corolla.median()
corolla.var()
corolla.skew()
plt.hist(corolla["Price"])
plt.hist(corolla["Age_08_04"])
plt.hist(corolla["KM"])                ###Data not normally distributed
plt.hist(corolla["HP"])
plt.hist(corolla["cc"])
plt.hist(corolla["Doors"])
plt.hist(corolla["Gears"])
plt.hist(corolla["Quarterly_Tax"])
plt.hist(corolla["Weight"])

plt.boxplot(corolla["Price"],0,"rs",0) 
plt.boxplot(corolla["Age_08_04"],0,"rs",0)
plt.boxplot(corolla["KM"],0,"rs",0)
plt.boxplot(corolla["HP"],0,"rs",0)
plt.boxplot(corolla["cc"],0,"rs",0)
plt.boxplot(corolla["Doors"],0,"rs",0)
plt.boxplot(corolla["Gears"],0,"rs",0)
plt.boxplot(corolla["Quarterly_Tax"],0,"rs",0)
plt.boxplot(corolla["Weight"],0,"rs",0)

corolla.isnull().sum() # no null values present

#data is not normal doing normalization
corolla=pd.DataFrame(preprocessing.normalize(corolla))

#renaming column names for convineance
corolla2=corolla.rename(index=str,columns={corolla.columns[0]: "price",corolla.columns[1]: "age",corolla.columns[2]:"km" , corolla.columns[3]:"hp",corolla.columns[4]:"cc",corolla.columns[5]:"doors",corolla.columns[6]:"gears",corolla.columns[7]:"tax",corolla.columns[8]:"weight"})
#sctterplot and histogram between variables
sns.pairplot(corolla2)
# Correlation matrix 
a=corolla2.corr()
### Splitting the data into train and test data 
cor_train,cor_test  = train_test_split(corolla2,test_size = 0.3) # 30% size
cor_train = cor_train.reset_index()
cor_train=cor_train.drop("index",axis=1)

# preparing the model on train data 
model_train = smf.ols("price~age+km+hp+cc+doors+gears+tax+weight",data=cor_train).fit()
model_train.summary()# all variable are significant
#all variable are significant so no need to check influence plot and no need to remove variables.

#calculating VIF's values of independent variables
rsq_km = smf.ols('km~age+hp+cc+doors+tax+weight+gears',data=cor_train).fit().rsquared  
vif_km = 1/(1-rsq_km) # 5.14

rsq_age = smf.ols('age~km+hp+cc+doors+tax+weight+gears',data=cor_train).fit().rsquared  
vif_age = 1/(1-rsq_age) # 3.070

rsq_wt = smf.ols('weight~km+hp+cc+doors+tax+age+gears',data=cor_train).fit().rsquared  
vif_wt = 1/(1-rsq_wt) # 203.77

rsq_tax = smf.ols('tax~km+hp+cc+doors+weight+age+gears',data=cor_train).fit().rsquared  
vif_tax = 1/(1-rsq_tax) # 3.79

rsq_hp = smf.ols('hp~km+tax+cc+doors+weight+age+gears',data=cor_train).fit().rsquared  
vif_hp = 1/(1-rsq_hp) # 22.72

rsq_cc = smf.ols('cc~km+hp+tax+doors+weight+age+gears',data=cor_train).fit().rsquared  
vif_cc = 1/(1-rsq_cc) # 58.80

rsq_dr = smf.ols('doors~km+hp+cc+tax+weight+age+gears',data=cor_train).fit().rsquared  
vif_dr = 1/(1-rsq_dr) # 6.35

rsq_gr = smf.ols('gears~km+hp+cc+doors+weight+age+tax',data=cor_train).fit().rsquared  
vif_gr = 1/(1-rsq_gr) # 121.83


################final model################
#here weight has max. VIF so remove and build model
model1 = smf.ols("price~age+km+hp+cc+doors+gears+tax",data=cor_train).fit()
model1.summary()# all variable are significant

# Added varible plot 
sm.graphics.plot_partregress_grid(model1)

# Predicted values of Profit
price_pred = model1.predict(cor_train)
price_pred

resid=price_pred-cor_train.price #av-fitted values
np.sum(resid)#sum of residuals nearly zero

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(cor_train.price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(price_pred,model1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

########    Normality plot for residuals ######
# histogram
plt.hist(model1.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)#normal

############ Homoscedasticity #######
# Residuals VS Fitted Values 
plt.scatter(price_pred,model1.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")

# train_data prediction
train_pred = model1.predict(cor_train)

# train residual values 
train_resid  = train_pred - cor_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))#0.0297

# prediction on test data set 
test_pred = model1.predict(cor_test)
# test residual values 
test_resid  = test_pred - cor_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))#0.0988
#Accurecy
train_pred.corr(cor_train.price)#0.9875
test_pred.corr(cor_test.price)#0.8975

'''
R-square =0.975
Accurecy=89.75%
This Model is nearly just right model, it can be more better if data will be more.

'''
