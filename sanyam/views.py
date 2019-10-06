from django.shortcuts import render, redirect, get_object_or_404, HttpResponse
from django.http import HttpResponseRedirect
from .forms import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit 
import cv2
import datetime
# Create your views here.

def main(request):
    if request.method == 'POST':
        # myfile = request.FILES['myfile']
        NORTH=request.POST.get('NORTH')
        SOUTH=request.POST.get('SOUTH')
        EAST=request.POST.get('EAST')
        WEST=request.POST.get('WEST')
        
        

        #img3 = cv2.imread(imgpath3, 1)
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


        data=pd.read_csv("//home//hold-on//Desktop//minor//Traffic_Counts_2012-2013.csv")


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


        
        densityn = 0
        densitys = 0
        densitye = 0
        densityw = 0

        path = "//home//hold-on//Desktop//minor//desktop//"

        imgpath1 =  path + "0.bmp"
        imgpath2 =  path + NORTH +".bmp"
        imgpath3 =  path + SOUTH +".bmp"
        imgpath4 =  path + EAST  +".bmp"
        imgpath5 =  path + WEST  +".bmp"
        #imgpath3 =  path + "Screenshot (56).png"


        img1 = cv2.imread(imgpath1, 1)
        img2 = cv2.imread(imgpath2, 1)
        img3 = cv2.imread(imgpath3, 1)
        img4 = cv2.imread(imgpath4, 1)
        img5 = cv2.imread(imgpath5, 1)
        #img3 = cv2.imread(imgpath3, 1)

        def fun(rgb):
            if rgb[0]>80:
                return False
            if rgb[1]>80:
                return False
            if rgb[2]>80:
                return False
            return True


        from numba import njit

        @njit
        def threshold_slow(image):
            def fun(rgb):
                #+if(rgb[0]<50 and rgb[1]<50 and rgb[2]>100):
                  #  return True
                if rgb[0]>80:
                    return False
                if rgb[1]>80:
                    return False
                if rgb[2]>80:
                    return False
                return True
            h = image.shape[0]
            w = image.shape[1]
            for y in range(0, h):
                for x in range(0, w):
                    if x<(int)(w/2):
                        image[y][x]=[0,0,0]
                        continue
                    if fun(image[y][x])==True:
                        image[y][x]=[0,255,0]

        threshold_slow(img1)
        threshold_slow(img2)
        threshold_slow(img3)
        threshold_slow(img4) 
        threshold_slow(img5)
        #threshold_slow(img3)





        imgf2 =cv2.subtract(img2,img1)
        imgf3 =cv2.subtract(img3,img1)
        imgf4 =cv2.subtract(img4,img1)
        imgf5 =cv2.subtract(img5,img1)
        #imgf =cv2.subtract(img3,img1)


        smoothing2=cv2.medianBlur(imgf2,5)
        smoothing3=cv2.medianBlur(imgf3,5)
        smoothing4=cv2.medianBlur(imgf4,5)
        smoothing5=cv2.medianBlur(imgf5,5)

        #smoothing3=cv2.medianBlur(imgf3,5)
        smoothing2 = cv2.cvtColor(smoothing2, cv2.COLOR_BGR2GRAY)
        smoothing3 = cv2.cvtColor(smoothing3, cv2.COLOR_BGR2GRAY)
        smoothing4 = cv2.cvtColor(smoothing4, cv2.COLOR_BGR2GRAY)
        smoothing5 = cv2.cvtColor(smoothing5, cv2.COLOR_BGR2GRAY)
        #smoothing3 = cv2.cvtColor(smoothing3, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('differ33.png', imgf3)



        #this was our first data ie least number of cars
        h = smoothing2.shape[0]
        w = smoothing2.shape[1]
        #defining num of divisions of img
        windowsize=76
        numofwindows=(int)(h/windowsize);
        if h%windowsize!=0:
            numofwindows+=1


        #calculating density
        densitiesofblock=[]
        tot=0
        #print(numofwindows)
        for x in range(0,numofwindows):
            densityofthisblock=0
            start = x*windowsize
            end = start + windowsize
            if end>h:
                end=h
            for y in range(start,end):
                for z in range(0,w):
                    tot+=1
                    if smoothing2[y][z]!=0:
                        densityofthisblock+=1
            densitiesofblock.append(densityofthisblock)

        #print(densitiesofblock)
        finaldensity = 0
        finaldensity+=((20000*densitiesofblock[0])+(10000*densitiesofblock[1])
                       +(5000*densitiesofblock[2])+(2000*densitiesofblock[3])
                       +(800*densitiesofblock[4])+(400*densitiesofblock[5])
                       +(100*densitiesofblock[6])+(50*densitiesofblock[7])
                       +(20*densitiesofblock[8])+(5*densitiesofblock[9])
                       +(1*densitiesofblock[10])) 
        densityn = finaldensity




        h = smoothing3.shape[0]
        w = smoothing3.shape[1]
        #defining num of divisions of img
        windowsize=76
        numofwindows=(int)(h/windowsize);
        if h%windowsize!=0:
            numofwindows+=1


        #calculating density
        densitiesofblock=[]
        tot=0
        #print(numofwindows)
        for x in range(0,numofwindows):
            densityofthisblock=0
            start = x*windowsize
            end = start + windowsize
            if end>h:
                end=h
            for y in range(start,end):
                for z in range(0,w):
                    tot+=1
                    if smoothing3[y][z]!=0:
                        densityofthisblock+=1
            densitiesofblock.append(densityofthisblock)

        #print(densitiesofblock)
        finaldensity = 0
        finaldensity+=((20000*densitiesofblock[0])+(10000*densitiesofblock[1])
                       +(5000*densitiesofblock[2])+(2000*densitiesofblock[3])
                       +(800*densitiesofblock[4])+(400*densitiesofblock[5])
                       +(100*densitiesofblock[6])+(50*densitiesofblock[7])
                       +(20*densitiesofblock[8])+(5*densitiesofblock[9])
                       +(1*densitiesofblock[10]))
        densitys = finaldensity





        h = smoothing4.shape[0]
        w = smoothing4.shape[1]
        #defining num of divisions of img
        windowsize=76
        numofwindows=(int)(h/windowsize);
        if h%windowsize!=0:
            numofwindows+=1


        #calculating density
        densitiesofblock=[]
        tot=0
        #print(numofwindows)
        for x in range(0,numofwindows):
            densityofthisblock=0
            start = x*windowsize
            end = start + windowsize
            if end>h:
                end=h
            for y in range(start,end):
                for z in range(0,w):
                    tot+=1
                    if smoothing4[y][z]!=0:
                        densityofthisblock+=1
            densitiesofblock.append(densityofthisblock)

        #print(densitiesofblock)
        finaldensity = 0
        finaldensity+=((20000*densitiesofblock[0])+(10000*densitiesofblock[1])
                       +(5000*densitiesofblock[2])+(2000*densitiesofblock[3])
                       +(800*densitiesofblock[4])+(400*densitiesofblock[5])
                       +(100*densitiesofblock[6])+(50*densitiesofblock[7])
                       +(20*densitiesofblock[8])+(5*densitiesofblock[9])
                       +(1*densitiesofblock[10]))
        densitye = finaldensity




        h = smoothing5.shape[0]
        w = smoothing5.shape[1]
        #defining num of divisions of img
        windowsize=76
        numofwindows=(int)(h/windowsize);
        if h%windowsize!=0:
            numofwindows+=1


        #calculating density
        densitiesofblock=[]
        tot=0
        #print(numofwindows)
        for x in range(0,numofwindows):
            densityofthisblock=0
            start = x*windowsize
            end = start + windowsize
            if end>h:
                end=h
            for y in range(start,end):
                for z in range(0,w):
                    tot+=1
                    if smoothing5[y][z]!=0:
                        densityofthisblock+=1
            densitiesofblock.append(densityofthisblock)

        #print(densitiesofblock)
        finaldensity = 0
        start = 1
        finaldensity+=((20000*densitiesofblock[0])+(10000*densitiesofblock[1])
                       +(5000*densitiesofblock[2])+(2000*densitiesofblock[3])
                       +(800*densitiesofblock[4])+(400*densitiesofblock[5])
                       +(100*densitiesofblock[6])+(50*densitiesofblock[7])
                       +(20*densitiesofblock[8])+(5*densitiesofblock[9])
                       +(1*densitiesofblock[10]))
        densityw = finaldensity


        #print(imgf2)
        print(cv2.imwrite(path + 'n.png', smoothing2))
        print(cv2.imwrite(path + 's.png', smoothing3))
        print(cv2.imwrite(path + 'e.png', smoothing4))
        print(cv2.imwrite(path + 'w.png', smoothing5))
        numofcarswest = funcalcars(densityw)
        numofcarsnorth= funcalcars(densityn)
        numofcarseast = funcalcars(densitye)
        numofcarssouth= funcalcars(densitys)
    #     print(numofcarswest)
    #     print(numofcarseast)
    #     print(numofcarsnorth)
    #     print(numofcarssouth)






        if(densityn==0 and densitys==0):
            finaltimer=0
        else:

            maxtimetimeron=120
            maxtimetimeron = min(maxtimetimeron,funcaltime(max(densityn,densitys)))
    #         print(maxtimetimeron)
    #         print("algo")
            mintimetimeron=0
            fulldensity = 64000000 #full congested lane allowed 
            mini =0
            maxi = 120
            while(int(mini)<int(maxi)):
                if (mini>maxi):
                    break
    #             print(mini,":",maxi)
                mid = (mini+maxi)/2
    #             print(mintimetimeron,"mintimeron")
                numofcaraccumulate = funcalusingdataset(mid)
    #             print(numofcaraccumulate,"cars")
                totcars = max(numofcarswest,numofcarseast) + numofcaraccumulate
    #             print(totcars,"cAAra")
    #             print(funcaldensityC(totcars),"density")
                if funcaldensityC(totcars) >= fulldensity :
                    maxi=mid
                else:
                    mini=mid
                    mintimetimeron=mid
            finaltimer = max(0,min(mintimetimeron,maxtimetimeron))
            print(finaltimer,"ANS")#answer 

    
    
    




        # ebd
        
        data = finaltimer
        if finaltimer==5:
            data="EXTEND THE PREVIOUS LIGHT BY 5 Sec"
        else:
            data="TIMER TO BE SET BY "+str(data)
            
        return render(request, 'cd.html',{'data':data})

    return render(request, 'ab.html')
    
def new(request):
    if request.method == 'POST':
        print('data received')
    else:
        print("not received")
    data="kdjfidj"
    return render(request, 'cd.html',{'data':data})

