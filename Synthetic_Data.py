import numpy as np
import functions
import krigingLE
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import math

#settings
iterations = 19

eBiasMag = [-3.22, -2.46]
eStdMag = [0.57,1.62]
#eStdMag = [0.,0.]




def maingraph(co, proxy, ind, gridwithoutsea, hyperbounds, a, lenfirstco, lons, 
         lats, parameters, fco, sco,tco,atemp, grid):
    rmses = []
    wiggles = []
    i = 0
    N=0.1
    while i< iterations:
        print(i)
        tempstations, temp = generatetemp(co, proxy, ind,N,gridwithoutsea) # create synthetic temp data
        model = krigingLE.fit(co,tempstations , proxy,proxy 
                              ,hyperbounds, verbose=0)
        yikrig = krigingLE.predictMean(model, gridwithoutsea)
        rmse = functions.rmse(temp, yikrig)
        wiggles.append(N)
        rmses.append(rmse)
        # if want to see the map every step
        #functions.kriging(co, tempstations, grid, proxy, proxy, hyperbounds, a , lenfirstco, lons, lats, parameters, fco, sco, tco, temp,N)
        N = N*1.3
        i+=1
    plt.figure(figsize=(15,10))
    plt.plot(wiggles,rmses, label = 'rmse')
    plt.xscale("log")
    plt.legend()
    plt.xlabel('Wiggles')
    plt.ylabel('rmse')
    plt.show()
    return wiggles, rmses


def mainmap(grid, hyperbounds1, hyperbounds12, hyperbounds123, lenfirstco, lons, 
         lats, parameters, fco, fproxy, co12 ,proxy12  ,co123 , proxy123, N,
         sco, tco, sproxy, tproxy,mask):
    
    #firstparty
    tempstations, temp = generatetemp(fco, fproxy, -1,N,grid) # create synthetic temp data
    functions.kriging(fco, tempstations, grid, fproxy, fproxy, 
                      hyperbounds1, 'first', lenfirstco, lons, lats, parameters, 
                      fco, sco, tco, temp,N, mask)
    """
    #second party
    tempstations, temp = generatetemp(sco, sproxy, 1,N,grid) # create synthetic temp data
    functions.kriging(sco, tempstations, grid, sproxy, sproxy, 
                      hyperbounds12, 'second', lenfirstco, lons, lats, parameters, fco, sco, tco, temp,N, mask)
    
    # third party 
    tempstations, temp = generatetemp(tco, tproxy, 1,N,grid) # create synthetic temp data
    functions.kriging(tco, tempstations, grid, tproxy, tproxy, 
                   hyperbounds12, 'third', lenfirstco, lons, lats, parameters, fco, sco, tco, temp,N)
    """
    #12party
    tempstations, temp = generatetemp(co12, proxy12, 0,N,grid) # create synthetic temp data
    functions.kriging(co12, tempstations, grid, proxy12, proxy12, 
                      hyperbounds12, 'first and second', lenfirstco, lons, lats, 
                      parameters, fco, sco, tco, temp,N, mask)
    """
    #123party
    tempstations, temp = generatetemp(co123, proxy123, 2 ,N,grid) # create synthetic temp data
    functions.kriging(co123, tempstations, grid, proxy123, proxy123, 
                      hyperbounds123, 'first, second and third', lenfirstco, 
                      lons, lats, parameters, fco, sco, tco, temp,N, mask)
    """
    PlotTrueMap(temp, lons, lats,N)


def generatetemp(co,proxy,ind,N,grid):
    amplitude = 5.36 # maxtemp - mintemp / 2 for 15 jan orsomething
    tempstations = amplitude *(np.cos(N*np.pi*co[:,0]) +np.cos(N*np.pi*co[:,1]))
    temp = amplitude*(np.cos(N*np.pi*grid[:,0]) +np.cos(N*np.pi*grid[:,1])) # truth
    if ind ==2:       
        for n in range(2): # add bias  and noise for 23 pd
            tempstations = tempstations + eBiasMag[n]*proxy[:,n]
            tempstations = tempstations + eStdMag[n]*np.multiply(proxy[:,n],np.random.randn(len(co),1))
    elif ind==0 or ind==1: # add bias and noise for 2 or 3pd
        tempstations = tempstations + eBiasMag[ind]*proxy[:,0]
        tempstations = tempstations + eStdMag[ind]*np.multiply(proxy[:,0],np.random.randn(len(co),1))
    return tempstations, temp

def PlotTrueMap(temp, lons, lats,N):
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
    tempmin = math.ceil(min(temp))-1
    tempmax = math.floor(max(temp))+1
    temp = np.reshape(temp,(100,100))
    # how many colours for colourbar
    ntemps = 7

    plt.title('True temperature', fontsize=45, pad = 20)
    levels = np.linspace(tempmin, tempmax,ntemps)
    kaart = ax.contourf(lons, lats, temp ,levels = levels, 
                        transform = ccrs.PlateCarree(), extend='both')
    cbar =plt.colorbar(kaart, pad=0.1)
    cbar.set_label( label='Celsius', size = 35, labelpad = 20)
    cbar.ax.tick_params(labelsize=35) 

    cbar.ax.tick_params(labelsize=25)
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
    ax.plot([], [], ' ', label="N = %.2f" %N)
    ax.legend(loc = 'upper left' , fontsize=30)
    plt.ylim([50.74, 53.55])
    plt.xlim([3.36, 7.23])
    plt.savefig('true', dpi=300, orientation="landscape", bbox_inches='tight')
    plt.show()

    