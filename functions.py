import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import krigingLE
import crossval
import pandas as pd
import math
from sklearn.metrics import mean_squared_error


#making temperature, location and proxy arrays
def alltempallcoprox(firstco, firsttemp, secondco, secondtemp, thirdco, thirdtemp):
    allco = []
    co23 = []
    co23 = np.concatenate((secondco,thirdco),axis=0)
    co13 = []
    co13 = np.concatenate((firstco,thirdco),axis=0)
    co12 = []
    co12 = np.concatenate((firstco,secondco),axis=0)
    allco = np.concatenate((co12,thirdco),axis=0)
    alltemp = []
    temp12 = []
    temp12 = np.concatenate((firsttemp,secondtemp),axis=0)
    temp13 = []
    temp13 = np.concatenate((firsttemp,thirdtemp),axis=0)
    alltemp = np.concatenate((temp12,thirdtemp),axis=0)
    Proxyfirst = np.matrix(np.zeros((len(firsttemp),1)))
    Proxysecond = np.matrix(np.ones((len(secondtemp),1)))
    Proxythird = np.matrix(np.ones((len(thirdtemp),1)))
    
    #estd en bias proxy
    Proxy12 = np.matrix(np.ones((len(temp12),1)))
    Proxy12[0:len(firsttemp)] = 0.0
    
    # estd en bias proxy
    Proxy13 = np.matrix(np.ones((len(temp13),1)))
    Proxy13[0:len(firsttemp)] = 0.0
    
    Proxy23 = np.matrix(np.ones((len(thirdtemp)+len(secondtemp),2)))
    Proxy23[0:len(secondtemp),1] = 0.0
    Proxy23[len(secondtemp):len(secondtemp)+len(thirdtemp),0] = 0.0

    Proxy = np.matrix(np.ones((len(alltemp),2)))
    Proxy[0:len(firsttemp),:] = 0.0
    Proxy[len(firsttemp):len(firsttemp)+len(secondtemp),1] = 0.0
    Proxy[len(firsttemp)+len(secondtemp):len(firsttemp)+len(secondtemp)+len(thirdtemp),0] = 0.0
    return allco, alltemp, co12, temp12, Proxyfirst, Proxysecond, Proxythird, Proxy, Proxy12, co13, temp13, Proxy13, co23, Proxy23

#read data and get it in right format
def readdata(data):
    data = pd.read_csv(data)
    df = data.iloc[:,0].str.split(';', expand=True)
    df.columns = data.columns[0].split(';')
    # pop rows with empty values 
    df = df.drop(df[df.avg_temp ==''].index)
    df = df.drop(df[df.longitude ==''].index)
    df = df.drop(df[df.latitude ==''].index)
    df['latitude'] = df['latitude'].astype(float)
    # take average of the location, some locations moved a bit during the day
    avg_lat = np.matrix(df.groupby(['station_id'])['latitude'].mean().to_numpy()).T
    df['longitude'] = df['longitude'].astype(float)
    avg_lon = np.matrix(df.groupby(['station_id'])['longitude'].mean().to_numpy()).T
    df['avg_temp'] = df['avg_temp'].astype(float)
    # average daily temp per station
    avg_temp = np.matrix(df.groupby(['station_id'])['avg_temp'].mean().to_numpy()).T
    coordinates = np.concatenate((avg_lon,avg_lat),axis=1)
    return coordinates, avg_temp

def maskarray(data,grid):
    data = pd.read_csv(data)
    data = data.iloc[:,0].str.split(';', expand=True).values
    mask = np.array([np.nan] * 100)
    i = 0
    while i <99:
        mask = np.concatenate((mask, data[i]), axis = 0)
        i+=1
    j = 0; k = 0
    mask = mask.astype(np.float) #otherwise 'nan'as string
    mask = np.reshape(mask,(100,100))
    mask = np.fliplr(mask)
    mask = np.rot90(mask,2)
    elementIndex = np.argwhere(np.isnan(np.reshape(mask,(10000,1))))[:,0]
    grid = list(grid)
    for j in range(0, len(elementIndex)):
        grid.pop(elementIndex[j]-j) # have to do minus j because index moves when you pop
    return np.array(grid), mask


def kriging (x,y,xi, eBiasProxy, eStdProxy, hyperbounds, a, nf, lon, lat, parameters, 
             fco, sco, tco, atemp,N, mask):
    global lats ; lats= lat; global lons; lons = lon; 
    global firstco; firstco = fco ;global secondco; secondco = sco;
    global thirdco; thirdco = tco; global alltemp; alltemp = atemp;  
    
    model = krigingLE.fit(x,y, eBiasProxy, eStdProxy,hyperbounds, parameters['verbose'])
    yikrig = krigingLE.predictMean(model,xi)
    Yikrig = np.reshape(yikrig,(parameters['ni'],parameters['ni']))
    ui = krigingLE.predictStd(model,xi)
    Ui = np.reshape(ui,(parameters['ni'], parameters['ni'])) # used for uncertainty
    averageunc = averageuncertainty(ui, mask)
    if parameters['uncertainty'] ==1:
        plot(0,1,0, a, Yikrig,x,y, Ui, alltemp, 
             averageunc,N,mask)
    if parameters['plotdifference'] ==1:
        plot(0,0,1, a, Yikrig,x,y, Ui, alltemp, 
             averageunc,N,mask)
   
def plot(plottemp, plotunc, plotdif, a , Yikrig, co, temp, Ui, alltemp, 
         averageunc, N, mask):
    plt.figure(figsize=(30, 15))
    ax = plt.axes(projection = ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 25}
    gl.ylabel_style = {'size': 25}
    tempmin = math.ceil(min(alltemp))-2
    tempmax = math.floor(max(alltemp))+2
    # min, max uncertainty
    uncmin = -3
    uncmax = 1

    # how many colours for colourbar
    ntemps = 7
    nlevels = 7

    # plot temp or uncertainty
    if plottemp ==1:
        plt.title('Temperature prediction', fontsize=45, pad = 20)
        levels = np.linspace(tempmin, tempmax,ntemps)
        kaart = ax.contourf(lons, lats, Yikrig ,levels = levels, 
                            transform = ccrs.PlateCarree(), extend='both')
        cbar =plt.colorbar(kaart, pad=0.1)
        cbar.set_label( label='Celsius', size = 35, labelpad = 20)
        cbar.ax.tick_params(labelsize=35) 
       
    if plotunc ==1:
        plt.title('Prediction uncertainty', fontsize=45, pad = 20)
        levels = np.linspace(uncmin, uncmax, nlevels)
        kaart = ax.contourf(lons,lats ,np.log10(Ui),levels = levels, 
                            cmap=plt.cm.BuGn, transform = ccrs.PlateCarree(), extend='both')
        cbar =plt.colorbar(kaart, pad=0.1)
        cbar.set_label( label='Celsius', size = 35, labelpad = 20)
        cbar.ax.set_yticklabels([np.round(10**t,decimals = 3) for t in cbar.ax.get_yticks()])
        cbar.ax.tick_params(labelsize=35)
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.02, 0.05, 'Average uncertainty is %.3f' %float(averageunc), transform=ax.transAxes, 
                color = 'black',fontsize=30, weight='bold', verticalalignment='top', bbox=props)
    
    if plotdif ==1:
        plt.title('Difference true and predicted temperature', fontsize=45, pad = 20)
        #levels = np.linspace(tempmin, tempmax,ntemps)
        levels = np.linspace(-3, 3,ntemps)
        truetemp = np.reshape(alltemp,(100,100))
        difference = Yikrig- truetemp
        avgdif, avgabsdif = averagedifference(difference,mask)
        kaart = ax.contourf(lons, lats, difference ,levels = levels, 
                            cmap=plt.cm.coolwarm, transform = ccrs.PlateCarree(),extend='both')
        cbar =plt.colorbar(kaart, pad=0.1)
        cbar.set_label( label='Celsius', size = 35, labelpad = 20)
        cbar.ax.tick_params(labelsize=35) 
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.02
                , 0.08, 'Average difference is %.3f' %float(avgdif) +'\nAverage absolute difference is %.3f' %float(avgabsdif)
                , transform=ax.transAxes, 
                color = 'black',fontsize=30, weight='bold', verticalalignment='top', bbox=props)
        
    # plotting the stations
    if a=='first, second and third':
        plt.scatter(np.array(firstco[:,0]),np.array(firstco[:,1]), marker = 's', s=130, c = 'black',
                    edgecolors='black' ,label="First party stations")#", %i" %len(firstco))
        plt.scatter(np.array(secondco[:,0]),np.array(secondco[:,1]), marker = 'o',s=110, c = 'w',
                    edgecolors='black',label="Second party stations")#", %i" %len(secondco))
        plt.scatter(np.array(thirdco[:,0]),np.array(thirdco[:,1]), marker = '^', s= 110,c = 'w',
                    edgecolors='black',label="Third party stations")#", %i" %len(thirdco))
    elif a =='first':
        plt.scatter(np.array(firstco[:,0]),np.array(firstco[:,1]), marker = 's', s=130, c = 'black',
                    edgecolors='black' ,label="first party stations")#", %i" %len(firstco))   
    elif a =='first and second':
        plt.scatter(np.array(firstco[:,0]),np.array(firstco[:,1]), marker = 's', s=130, c = 'black',
                    edgecolors='black' ,label="first party stations")#", #stations =%i" %len(firstco))
        plt.scatter(np.array(secondco[:,0]),np.array(secondco[:,1]), marker = 'o',s = 110, c = 'w',
                    edgecolors='black',label="second party stations")#", #stations =%i" %len(secondco))
    
    elif a =='first and third':
        plt.scatter(np.array(firstco[:,0]),np.array(firstco[:,1]), marker = 's', s=130, c = 'black',
                    edgecolors='black' ,label="first party stations, %i" %len(firstco))
        plt.scatter(np.array(thirdco[:,0]),np.array(thirdco[:,1]), marker = 'o',s = 110, c = 'w',
                    edgecolors='black',label="third party stations, %i" %len(thirdco))                   
    else:
        plt.scatter(np.array(co[:,0]),np.array(co[:,1]), marker = '^', s = 110 ,c = 'w', 
                    edgecolors='black', label = "stations, %i" %len(co))

    cbar.ax.tick_params(labelsize=25)
    if N>0:
        ax.plot([], [], ' ', label="N = %.2f" %N)
    ax.legend(loc = 'upper left' , fontsize=30)
    #params for plotting borders/rivers/ocean
    resol = '10m'  # use data at this scale
    bodr = cartopy.feature.NaturalEarthFeature(category='cultural', 
        name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
    ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
        scale=resol, edgecolor='none', facecolor='none')
    oceanblue =  cartopy.feature.NaturalEarthFeature('physical', 'ocean', \
        scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water']) # use this instead of ocean to mask sea.
    lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', \
        scale=resol, edgecolor='b', facecolor='none') # if we want to plot lakes, facecolor=cfeature.COLORS['water']
    #rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', \scale=resol, edgecolor='b', facecolor='none')
    lakesblue = cartopy.feature.NaturalEarthFeature('physical', 'lakes', \
        scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
        
    # plotting features
    ax.add_feature(oceanblue, linewidth=0.5)
    ax.add_feature(lakesblue, linewidth=0.5)
    ax.add_feature(ocean, linestyle='-',linewidth = 2, edgecolor='k', alpha=1)
    ax.add_feature(lakes, linestyle='-', linewidth = 2, edgecolor='k', alpha=1)
    #ax.add_feature(rivers, linewidth=1)
    ax.add_feature(bodr, linestyle='-', linewidth = 4, edgecolor="#cccccc", alpha=1)
    plt.ylim([50.74, 53.55])
    plt.xlim([3.36, 7.23])

    if plottemp==1:
        plt.savefig(str(a)+'pd'+'temp.png', dpi=300, orientation="landscape", bbox_inches='tight')
    elif plotunc==1:
        plt.savefig(str(a)+'pd'+'uncertainty.png', dpi=300, orientation="landscape", bbox_inches='tight')
    else:
        plt.savefig(str(a)+'pd'+'tempzl.png', dpi=300, orientation="landscape", bbox_inches='tight')
    plt.show()


def averageuncertainty(ui, mask):
    uicopy = ui.copy()
    uicopy = uicopy.reshape(100,100)
    uicopy[np.isnan(mask)] = 0
    np.square(uicopy)
    averageui = uicopy[np.nonzero(uicopy)].mean()
    return averageui

def averagedifference(difference, mask):
    difference[np.isnan(mask)] = 0
    averagediff = difference[np.nonzero(difference)].mean()
    averageabsdiff = abs(difference[np.nonzero(difference)]).mean()
    return averagediff, averageabsdiff


def rmse(measurement,prediction):    
    prediction = np.array(prediction, dtype=float)
    measurement = np.array(measurement, dtype=float)
    rms = np.sqrt(mean_squared_error(prediction,measurement, 
                             multioutput='raw_values', squared=False)) #(y_real, y_predicted, squared=false)
    return rms[0]


        


