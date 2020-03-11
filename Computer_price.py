import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the File

computer=pd.read_csv("E:\\ExcelR\\Python codes and python datasets\\Assignments in Python\\Assignment 2- Multilinear Regression\\Assignment 9 - Multiple Linear Regression.\\Computer_Data.csv")

# EDA
computer.columns

plt.hist(computer.price)#Output variable follows some sort of normal distribution
plt.hist(computer.speed)# Not a normal distribution 
plt.hist(computer.hd)
plt.hist(computer.ram)
plt.hist(computer.screen)
plt.hist(computer.cd) # Need to create dummy variable
plt.hist(computer.multi) # Need to create dummy variable
plt.hist(computer.premium) # Need to create dummy variable
plt.hist(computer.ads)# Need to check in the coorelation whether we have to include this variable in the model
plt.hist(computer.trend)# Need to create dummy variable 

# understanding the corellation between variables

import seaborn as sn 
sn.pairplot(computer) # does not make much sense

computer=computer.drop("Unnamed: 0",axis="columns")
# creating dummies for the CD column
dummies_cd=pd.get_dummies(computer.cd)
dummies_cd=dummies_cd.rename(columns={"yes":"cdyes","no":"cdno"})
# Concatinating the dummy cd and removing the orginal cd
computer=pd.concat([computer,dummies_cd],axis="columns")
computer=computer.drop("cd",axis="columns")

# creating dummy foor multi
dummies_multi=pd.get_dummies(computer.multi)
dummies_multi=dummies_multi.rename(columns={"yes":"multiyes","no":"multino"})
dummies_multi
computer=pd.concat([computer,dummies_multi],axis="columns")

dummies_premium=pd.get_dummies(computer.premium)
dummies_premium=dummies_premium.rename(columns={"yes":"premiumyes","no":"premiumno"})
computer=pd.concat([computer,dummies_premium],axis="columns")
# droping multi and premium from the table
computer=computer.drop("multi",axis="columns")
computer=computer.drop("premium",axis="columns")

# checking whether ads and trend columns are important 

np.corrcoef(computer.ads,computer.price) # Very low corellation so need not consider
np.corrcoef(computer.trend,computer.price)# Very low corellation so need not consider
computer=computer.drop("ads",axis="columns")
computer=computer.drop("trend",axis="columns")

# Final dataframe
computer.columns
import seaborn as sns
sns.pairplot(computer)
# Dummy variable has no impact at all we need not consider them for the model
# Buildng the model
import statsmodels.formula.api as smf
ml_1=smf.ols("price~speed+hd+ram+screen+ads+trend+cdno+cdyes+multino+multiyes+premiumno+premiumyes",data=computer).fit()
ml_1.summary()# r2=0.77 but p values is not acceptable

ml_2=smf.ols("price~speed+hd+ram+screen+ads+trend",data=computer).fit()
ml_2.summary() # rsquared 71% all p values acceptable

ml_3=smf.ols("np.log(price)~speed+hd+ram+screen+ads+trend",data=computer).fit()
ml_3.summary()# 72%

ml_4=smf.ols("np.log(price)~np.log(speed)+np.log(ads)+np.log(trend)+np.log(hd)+np.log(ram)+np.log(screen)",data=computer).fit()
ml_4.summary()#70%

ml_5=smf.ols("np.invert(price)~speed+hd+ram+ads+trend+screen",data=computer).fit()
ml_5.summary()#71%

#ploting an av plot showing the influence of each independent variable on dependent variable

import statsmodels.api as sm
sm.graphics.plot_partregress_grid(ml_3)






