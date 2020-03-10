import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the csv file
startup=pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 2- Multilinear Regression\\Assignment 9 - Multiple Linear Regression\\50_Startups.csv")
startup
# We have to create dummy variables for state

dummies=pd.get_dummies(startup.State)
dummies
startup_new=pd.concat([startup,dummies],axis="columns")
startup_new
startup_new=startup_new.drop("State",axis="columns")
startup_new

# Corellation
startup_new.corr()

import seaborn as sns
sns.pairplot(startup_new)

# Building the First Model

import statsmodels.formula.api as smf
# We have to change the column name as it as spaces the model function is throwing error
startup_new.columns
startup_new=startup_new.rename(columns={"R&D":"Research"})

# Model with dummy variables
ml_1=smf.ols("Profit~(Research+Admin+Marketing+California+Florida+NewYork)",data=startup_new).fit()
ml_1.summary() # Problem with Admin and marketing

# Model without dummy variables

ml_2=smf.ols("Profit~Research+Admin+Marketing",data=startup_new).fit()
ml_2.summary() # Dummy variables do not have that much of an impact in this model

# P- values are not acceptable for Admin and marketing 

# creating a model with Admin only

ml_admin=smf.ols("Profit~Admin",data=startup_new).fit()
ml_admin.summary() # Rsquared value very low and p value is >0.05 for admin

# creating a model with marketing only
ml_marketing=smf.ols("Profit~Marketing",data=startup_new).fit()
ml_marketing.summary()# 0.55 and p value is acceptable

# Creating a model with admin and marketing
ml_mar_adm=smf.ols("Profit~Admin+Marketing",data=startup_new).fit()
ml_mar_adm.summary()# 0.61 and p value is not acceptable

# Identifying any problamatic observation

import statsmodels.api as sm
sm.graphics.influence_plot(ml_2) # Influential records- 49,48,46 & 45

startup_new=startup_new.drop(startup_new.index[[49,48,46,45]],axis=0)

ml_inf=smf.ols("Profit~Research+Admin+Marketing",data=startup_new).fit()
ml_inf.summary() # p- value is not acceptable for admin and marketing

# Checking whether there is a collenearity issue by calculating VIF for input variables
rsq_res=smf.ols("Research~Admin+Marketing",data=startup_new).fit().rsquared
VIF_res=1/(1-rsq_res)

rsq_admin=smf.ols("Admin~Research+Marketing",data=startup_new).fit().rsquared
VIF_admin=1/(1-rsq_admin)

rsq_marketing=smf.ols("Marketing~Research+Admin",data=startup_new).fit().rsquared
VIF_marketing=1/(1-rsq_marketing)

# putting the VIF values in a dataframe

d1={"Variables":["Research","Admin","Marketing"],"VIF":[VIF_res,VIF_admin,VIF_marketing]}
VIF_Frame=pd.DataFrame(d1)
VIF_Frame # there do not exist any collenearity issue

# AV Plot 
import statsmodels.api as sm
sm.graphics.plot_partregress_grid(ml_2)

# Removing the admin variable and creating the model

model=smf.ols("Profit~Research+Marketing",data=startup_new).fit()
model.summary() # 0.96 & P- value is acceptable

# Graph showing the influence of input varaiables on output variable

import statsmodels.api as sm
sm.graphics.plot_partregress_grid(model)


