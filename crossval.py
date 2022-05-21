import numpy as np
import krigingLE
import math


# use these to filter out 1pd that is in the sea.
statlonmin = 3.3
statlonmax = 7.3
statlatmin = 50.7
statlatmax = 53.5

def unison_shuffled_copies(stat, temp):
    p = np.random.permutation(len(temp))
    return np.array(stat[p]) ,np.array(temp[p])

def locationfilter(stat, temp, prox,nf): # filter out 1pd thats in the sea
    i = 0
    while i< len(stat):
        if statlonmin<stat[i][0]<statlonmax and statlatmin < stat[i][1]< statlatmax:
            i+=1
        else:
            stat = np.delete(stat,[i], axis=0)
            temp = np.delete(temp,[i], axis=0)
            prox = np.delete(prox,[i], axis=0)
            nf+=-1
    return stat, temp, prox,nf

def makinggroups(stat, temp, prox, gs, fng): #gs is groupsize
    fstatgroups = []
    ftempgroups = []
    fproxgroups = []
    i = 0
    while i < fng:
        stati = []
        tempi = []
        proxi = []
        j = 0
        while j< gs:
            if (i*gs+j)<len(stat): # do this because of rounding gs
                stati.append(np.array(stat)[i*gs+j])
                tempi.append(np.array(temp)[i*gs+j])
                proxi.append(np.array(prox)[i*gs+j])            
            j+=1
        fstatgroups.append(stati)
        ftempgroups.append(tempi)
        fproxgroups.append(proxi)
        i+=1
    return fstatgroups, ftempgroups, fproxgroups


def dividingdata(stat, temp, prox,nf, fng): # putting knmi data in different arrays
    firststat = []
    firsttemp = []
    firstprox = []
    otherstat = []
    othertemp = []
    otherprox = []
    j = 0
    while j< nf:
        firststat.append((np.array(stat)[j]))
        firstprox.append(np.array(prox)[j])
        firsttemp.append(np.array(temp)[j])
        j+=1
    while j<len(stat):
        otherprox.append(np.array(prox)[j])
        otherstat.append((np.array(stat)[j]))
        othertemp.append(np.array(temp)[j]) 
        j+=1
    firststat, firsttemp, firstprox , nf= locationfilter(np.array(firststat), 
                                                         np.array(firsttemp),
                                                         np.array(firstprox),nf)
    firststat, firsttemp = unison_shuffled_copies(firststat, firsttemp)
    fgs = round(len(firststat)/fng) # 1pd data groupsize
    fstatgroups, ftempgroups, fproxgroups = makinggroups(firststat, firsttemp, firstprox, fgs, fng)
    return np.array(otherprox), np.array(otherstat), np.array(othertemp), fstatgroups, ftempgroups, fproxgroups, nf
 
   
def comperror(fstatgroups, ftempgroups, fproxgroups, otherstat, othertemp, otherprox, hyperbounds, nf, fng):
    i = 0
    error = 0
    error_theoretical = 0
    while i< fng: #Pick 1 1pd group for your test group  # fng
        teststat = fstatgroups[i] # set picked group as test data
        testtemp = ftempgroups[i] 
        if len(teststat)>0: #need this if statement due to rounding of groupsize
            if len(otherstat)!=0:
                trainingstat = otherstat.copy() #2/3 pd always trainingdata
                trainingtemp = othertemp.copy()
                trainingprox = otherprox.copy()
                j = 0
                while j < fng: # add the other 1pd groups to the training group
                    if j == i:
                        if (j+1)<fng:
                            j+=1
                    if len(fstatgroups[j])>0:
                        trainingstat = np.concatenate((trainingstat, fstatgroups[j]), axis = 0)
                        trainingtemp = np.concatenate((trainingtemp, ftempgroups[j]), axis = 0)
                        trainingprox = np.concatenate((trainingprox, fproxgroups[j]), axis = 0)
                        j+=1
                    else:
                        j = fng
            else: #first party only
                tstat = np.delete(fstatgroups.copy(),[i], axis=0)
                ttemp = np.delete(ftempgroups.copy(),[i], axis=0)
                tprox = np.delete(fproxgroups.copy(),[i], axis=0)
                trainingstat = []
                trainingtemp= []
                trainingprox = []
                j = 0
                while j<(fng-1):
                    k = 0
                    while k< len(tstat[j]):
                        trainingstat.append(tstat[j][k])
                        trainingtemp.append(ttemp[j][k])
                        trainingprox.append(tprox[j][k])
                        k+=1
                    j+=1         
            model = krigingLE.fit(np.array(trainingstat),np.array(trainingtemp) ,
                                  np.array(trainingprox ), np.array(trainingprox) 
                                  ,hyperbounds, verbose=0)
            yikrig = krigingLE.predictMean(model, np.array(teststat))
            uikrig = krigingLE.predictStd(model,np.array(teststat))
            k = 0
            while k<len(testtemp):
                error += (float(yikrig[k])-float(testtemp[k]))**2
                if not math.isnan(uikrig[k]):
                    error_theoretical += uikrig[k]**2
                k+=1
            i+=1
        else:
            i = fng          
    finalerror = np.sqrt(error/nf) # error/ #firstparty stations
    finalerror_theoretical = np.sqrt(error_theoretical/nf)
    return finalerror, finalerror_theoretical

def crossval(stat, temp, prox, nf, bias, theta, noise,xi, lonmin, lonmax, latmin, latmax,nc, fng):
    hyperbounds = {"lb_bias": tuple(bias), "ub_bias":tuple( bias), "lb_noise" :tuple(noise)
              , "ub_noise":tuple(noise), "lb_theta":tuple(theta),"ub_theta": tuple(theta)}
    i = 0
    errorsum = 0
    theoretical_errorsum = 0
    while i < nc:
        otherprox, otherstat, othertemp, fstatgroups, ftempgroups, fproxgroups, number1ps = dividingdata(stat, temp, prox,nf, fng)

        error, error_theoretical = comperror(fstatgroups, ftempgroups,
                                             fproxgroups, otherstat, othertemp, otherprox, hyperbounds, number1ps, fng)
        errorsum+=error
        theoretical_errorsum+= error_theoretical  
        i+=1
    finalerror = errorsum/nc
    finalerror_theoretical = theoretical_errorsum/nc
    print('error',finalerror)
    print('theoretical error', finalerror_theoretical)
    return finalerror, finalerror_theoretical