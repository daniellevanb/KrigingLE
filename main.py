import numpy as np
import functions
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
start_time = time.time()

parameters = {"uncertainty":(1), # 0 or 1 indicates wether to plot uncertainty map
              "temp":(1), # '' temperature map
              "plotdifference":(1),
              "verbose":(0),
              "crossvalidation":(1), #'' perform cross validation
              "nc":(1), # how many times perform cross validation
              "fng":(5), # how many cross val test groups
              "howmanyiterations":(5), # how many times comput error/price function before median
              "ni":(100),
              "eBiasMag": ([-.3,1.]),
              "lonmin":(3.36), # plotting boundaries
              "lonmax":(7.23),
              "latmin":(50.74),
              "latmax": (53.55)
              } 


lons = np.linspace(parameters['lonmin'], parameters['lonmax'], parameters['ni'])
lats = np.linspace(parameters['latmin'], parameters['latmax'], parameters['ni'])
Xi, Yi = np.meshgrid(lons, lats)
xi = Xi.reshape(parameters['ni']*parameters['ni'],1)
yi = Yi.reshape(parameters['ni']*parameters['ni'],1)
grid = np.concatenate((xi,yi), axis = 1)
# hyperbounds for 1,12,123 pd, for now fixed for faster computation 
hyperbounds1 ={"lb_bias": (0.), "ub_bias": (0.), "lb_noise" :(0.1)
          , "ub_noise":(0.1), "lb_theta":(.01,.01),"ub_theta": (10.,10.)}

hyperbounds2 ={"lb_bias": (-4.), "ub_bias": (4.), "lb_noise" :(.1)
          , "ub_noise":(5.), "lb_theta":(.01,.01),"ub_theta": (1.,1.)}


hyperbounds12 = {"lb_bias": (-4.), "ub_bias": (4.), "lb_noise" :(0.1)
          , "ub_noise":(5.), "lb_theta":(.01,.01),"ub_theta": (10.,10.)}

hyperbounds123 = {"lb_bias": (-4., -4.), "ub_bias": (4., 4.), "lb_noise" :(0.1, 0.1)
          , "ub_noise":(5.,5.), "lb_theta":(.01,.01),"ub_theta": (10., 10.)}


first = open('C:\\Users\\dvanb\\OneDrive\\Documenten\\scriptie\\syntheticdata\\data\\1pd_20190125.csv')
second = open('C:\\Users\\dvanb\\OneDrive\\Documenten\\scriptie\\syntheticdata\\data\\2pd_20190125.csv')
third = open('C:\\Users\\dvanb\\OneDrive\\Documenten\\scriptie\\syntheticdata\\data\\3pd_20190125.csv')
mask = open('C:\\Users\\dvanb\\OneDrive\\Documenten\\scriptie\\syntheticdata\\data\\NL_Mask_100x100.csv')



########getting data and putting it in right format#########
firstco, firsttemp = functions.readdata(first)
secondco , secondtemp= functions.readdata(second)
thirdco, thirdtemp = functions.readdata(third)
gridNL, mask = functions.maskarray(mask, grid)



allco, alltemp, co12, temp12, Proxyfirst, Proxysecond, Proxythird, Proxy, Proxy12, co13, temp13, Proxy13, co23, Proxy23 = functions.alltempallcoprox(firstco, firsttemp, secondco, secondtemp, thirdco, thirdtemp)
# Use for reduced kriging


###### kriging +plots#########

"""
print('1')
functions.kriging (firstco ,firsttemp ,grid, Proxyfirst, 
                    Proxyfirst, hyperbounds1, 'first', len(firstco), 
                    lons, lats, parameters, firstco,secondco, 
                    thirdco,alltemp, 0, mask)
"""
print('2')
functions.kriging (secondco ,secondtemp ,grid, Proxysecond, 
                    Proxysecond, hyperbounds2, 'second', len(firstco), 
                    lons, lats, parameters, firstco,secondco, 
                   thirdco,alltemp, 0, mask)
"""
print('3')
functions.kriging (thirdco ,thirdtemp ,grid, Proxythird, 
                    Proxythird, hyperbounds2, 'third', len(firstco), 
                    lons, lats, parameters, firstco,secondco, 
                   thirdco,alltemp, 0,  mask)

print('12')
functions.kriging (co12 ,temp12 ,grid, Proxy12, 
                    Proxy12, hyperbounds12, 'first and second', len(firstco), 
                    lons, lats, parameters, firstco,secondco, 
                   thirdco,alltemp,0,  mask)

print('13')
functions.kriging (co13 ,temp13 ,grid, Proxy13, 
                    Proxy13, hyperbounds12, 'first and third', len(firstco), 
                    lons, lats, parameters, firstco, secondco, 
                    thirdco,alltemp,0, mask,gridNL)

print('123')
functions.kriging (allco , alltemp ,grid, Proxy, 
                    Proxy, hyperbounds123, 'first, second and third', len(firstco), 
                    lons, lats, parameters, firstco,secondco, 
                    thirdco, alltemp,0,  mask)


##Synthetic data ####
# produces maps for 1pd, 12pd and 123pd given N
N=1.5
PlanB.mainmap(grid, hyperbounds1, hyperbounds12, hyperbounds123, len(firstco), 
              lons, lats, parameters, firstco, Proxyfirst, co12, Proxy12, allco, 
              Proxy, N, secondco, thirdco, Proxysecond, Proxythird, mask)


#RMSE graphs
print('1')
wiggles1 , rmses1 = PlanB.maingraph(firstco,Proxyfirst,-1, gridNL, hyperbounds1 , 'first', len(firstco),
           lons, lats, parameters, firstco, secondco, thirdco, alltemp, grid) #-1 indicates no noise/bias adding necessary

print('2')
wiggles2, rmses2  =PlanB.maingraph(secondco, Proxysecond,0, gridNL, hyperbounds2 , 'second', len(firstco),
            lons, lats, parameters, firstco, secondco, thirdco, alltemp, grid)

print('3')
wiggles3, rmses3  =PlanB.maingraph(thirdco, Proxythird,1, gridNL, hyperbounds2 , 'third', len(firstco),
            lons, lats, parameters, firstco, secondco, thirdco, alltemp, grid)

print('12')
wiggles12, rmses12 = PlanB.maingraph(co12,Proxy12,0, gridNL, hyperbounds12 , 'first and second', len(firstco),
          lons, lats, parameters, firstco, secondco, thirdco, alltemp, grid)

print('23')
wiggles23, rmses23 = PlanB.maingraph(co23,Proxy23,2, gridNL, hyperbounds123 , 'second and third', len(firstco),
          lons, lats, parameters, firstco, secondco, thirdco, alltemp, grid)

print('13')
wiggles13, rmses13  = PlanB.maingraph(co13, Proxy13,1, gridNL, hyperbounds12 , 'first and third', len(firstco),
            lons, lats, parameters, firstco, secondco, thirdco, alltemp, grid)

print('123')
wiggles123, rmses123= PlanB.maingraph(allco,Proxy,2, gridNL, hyperbounds123 , 'first, second and third', len(firstco),
           lons, lats, parameters, firstco, secondco, thirdco, alltemp, grid)

plt.figure(figsize=(10, 6))
plt.plot(wiggles1, rmses1, label = "1PD", color = 'blue')

plt.plot(wiggles2, rmses2, label = "2PD", color = 'red')
plt.plot(wiggles3, rmses3, label = "3PD", color = 'gold')
#plt.plot(wiggles23, rmses23, label = "23PD", color = 'orange')
plt.plot(wiggles12, rmses12, label = "12PD", color = 'purple')
plt.plot(wiggles13, rmses13, label = "13PD", color = 'green')
plt.plot(wiggles123, rmses123, label = "123PD", color = 'brown')

plt.grid()
ax = plt.axes()
ax.set_facecolor("gainsboro")
plt.title("Fitting performance based on weather complexity")
plt.xscale("log")
plt.xlabel("Oscillation per degree")
plt.ylabel('RMSE in Celsius')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig('rmse', dpi=500, orientation="landscape", bbox_inches='tight')
plt.show()

"""
print ("My program took", (time.time() - start_time)/60, " minutes to run")

