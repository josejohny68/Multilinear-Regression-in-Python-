import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the file
corolla=pd.read_excel("E:\\ExcelR\\Assignments\\Assignment 9 - Multiple Linear Regression\\ToyotaCorolla.xlsx")
corolla.columns
corolla=corolla.drop("Id",axis="columns")
corolla=corolla.drop(["Model","Mfg_Month"],axis="columns")
corolla=corolla.drop(['Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders','Mfr_Guarantee','BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer','CD_Player','Central_Lock','Powered_Windows','Power_Steering','Radio','Mistlamps','Sport_Model','Backseat_Divider','Metallic_Rim','Radio_cassette','Tow_Bar'],axis="columns")
# Data set as been trimmed as per the question
corolla.corr() # From the table I dont see any variable creating the issue of collinearity

import seaborn as sns
sns.pairplot(corolla)

# Building the First Model

import statsmodels.formula.api as smf
model1=smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=corolla).fit()
model1.summary()#0.86 but p values not acceptable for cc and doors

# ploting the graph for influence records
import statsmodels.api as sms
sms.graphics.influence_plot(model1) # Issue with the 80th record

corolla=corolla.drop(corolla.index[[80]],axis=0)

model2=smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=corolla).fit()
model2.summary()# 0.86 but p value of doors not acceptable

# Drawing AV plot to check if doors have an impact on the output variable

import statsmodels.api as sms
sms.graphics.plot_partregress_grid(model2)
# Doors has no imapct so we can remove doors and build the model

model=smf.ols("Price~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight",data=corolla).fit()
model.summary() # 86%

# influence of each input variable on the output variable
sms.graphics.plot_partregress_grid(model)



