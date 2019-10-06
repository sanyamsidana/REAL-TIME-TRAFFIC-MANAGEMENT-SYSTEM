
# coding: utf-8

# In[19]:


import numpy as np
import datetime
import pandas as pd
dtypes = [('density', int), ('car', int)]
## values to be put in array
values = [(4153157,6), (17439146,15), (34800863,22), (50018314,27),(65703794,35)]
## creating array
arr = np.array(values, dtype = dtypes)
np.savetxt('density2car.csv', arr, delimiter=',', header="density,car")
# arr2=[]
# np.loadtxt('density2car.csv')


# In[5]:


dtypes = [('density', int), ('time', int)]
## values to be put in array
values = [(4153157,8), (17439146,15), (34800863,22), (50018314,26),(65703794,31)]
## creating array
arr = np.array(values, dtype = dtypes)
np.savetxt('density2time.csv', arr, delimiter=',', header="density,time")


# In[6]:


dtypes = [('time', int), ('density', int)]
## values to be put in array
values = [(8,4153157), (15,17439146), (22,34800863), (26,50018314),(31,65703794)]
## creating array
arr = np.array(values, dtype = dtypes)
np.savetxt('time2density.csv', arr, delimiter=',', header="time,density")


# In[7]:


import pandas as pd
car2density=pd.read_csv('density2car.csv')
car2density.columns=["density","car"]
car2density0=np.array(car2density["car"])
car2density1=np.array(car2density["density"])


# In[8]:


density2car=pd.read_csv('density2car.csv')
density2car.columns=["density","car"]
density2car0=np.array(density2car["density"])
density2car1=np.array(density2car["car"])


# In[9]:


density2time=pd.read_csv('density2time.csv')
density2time.columns=["density","time"]
density2time0=np.array(density2time["density"])
density2time1=np.array(density2time["time"])


# In[10]:


time2density=pd.read_csv('time2density.csv')
time2density.columns=["time","density"]
time2density0=np.array(time2density["time"])
time2density1=np.array(time2density["density"])


# In[11]:


import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
density2car0=density2car0.reshape(-1,1)
density2time0=density2time0.reshape(-1,1)
time2density0=time2density0.reshape(-1,1)
car2density0=car2density0.reshape(-1,1)

algcar = linear_model.LinearRegression()
algcar.fit(density2car0,density2car1)
algtime = linear_model.LinearRegression()
algtime.fit(density2time0,density2time1)
algdensityt = linear_model.LinearRegression()
algdensityt.fit(time2density0,time2density1)


algdensityc = linear_model.LinearRegression()
algdensityc.fit(car2density0,car2density1)

# print(algcar.predict(37000000))
# print(algtime.predict(37000000))
# print(algdensity.predict(22))


# In[12]:


data=pd.read_csv("Traffic_Counts_2012-2013.csv")


# In[14]:


data.drop('From', axis=1, inplace=True)
data.drop('Segment ID', axis=1, inplace=True)
data.drop('To', axis=1, inplace=True)
data.drop('Direction', axis=1, inplace=True)
data.drop('Date', axis=1, inplace=True)
data.drop('Roadway Name', axis=1, inplace=True)


# In[15]:


data.columns=["ID","0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]


# In[16]:


one = data['ID'] ==1
two = data['ID'] ==2
data_one=data[one]
data_two=data[two]


# In[18]:


data_one.mean()


# In[35]:


def funcalcars(density):
    return int(algcar.predict(density)[0])
def funcaltime(density):
    return int(algtime.predict(density)[0])
def funcaldensityC(cars):
    return int(algdensityc.predict(cars)[0])
def funcalusingdataset(timer,ID=1):
    z=str(datetime.datetime.now())
    c=int(z[11])*10+int(z[12])
    if ID==1:
        hour=data_one.mean()[c]
        sec=hour/(60*10)
        return int(timer*sec)+1    
    
    

# print(algcar.predict(37000000))
# print(algtime.predict(37000000))
# print(algdensity.predict(22))


# In[36]:




