import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

regulizer = 0.00001 # for small eigenvalues of matrix A

def strd(x):
    out = str(np.round(x, decimals = 3))
    return(out)

def fit(x,y,eBiasProxy,eStdProxy ,hyperbounds, verbose=0): 
    # normalise data
    ndim = x.shape[1]
    mu = np.mean(y)
    st = np.std(y)
    yn = (y-mu)/st
    ebian = eBiasProxy/st
    nebia = ebian.shape[1]
    estdn = eStdProxy/st
    #nestd = max(estdn.shape[1],1)
    if verbose==1:
        print('> data summary')
        print('  > number of dimensions = '+str(ndim))
        print('  > number of samples = '+str(y.shape[0]))
        print('  > coordinates x mean = '+strd(np.mean(x,axis=0))) 
        print('  > coordinates x std = '+strd(np.std(x,axis=0)))
        print('  > data y mean = '+strd(mu)) 
        print('  > data y std = '+strd(st))
        print('  > number of error bias proxies = '+str(nebia))
        print('  > number of error std  proxies = '+str(estdn.shape[1]))
        if estdn.shape[1]==0:
            print('    > 1 global error std proxy added by default')
    # set defaults
    if verbose == 1:
        print('> setting defaults')     
    if estdn.shape[1] == 0:
        estdn = 0*yn + 0.01
    # make lower/upperbound arrays
    lowerbounds = []
    for item in ('lb_bias', 'lb_noise', 'lb_theta'):
        if item == 'lb_bias':
            if isinstance(hyperbounds['lb_bias'], float):
                lowerbounds.append(hyperbounds[item])
            else:
                lowerbounds+=list(hyperbounds[item])         
        elif item == 'lb_noise':
            if isinstance(hyperbounds['lb_noise'], float):
                if hyperbounds['lb_noise']!=0:
                    lowerbounds.append(np.log(hyperbounds[item]))
                else:
                    lowerbounds.append(0)
            else:
                lowerbounds+=list(np.log(hyperbounds[item]))
       
        elif item =='lb_theta':
            lowerbounds+=list(np.log(hyperbounds[item]))

    upperbounds = []
    for item in ('ub_bias', 'ub_noise', 'ub_theta'):
        if item == 'ub_bias':
            if isinstance(hyperbounds['ub_bias'], float):
                upperbounds.append(hyperbounds[item])
            else:
                upperbounds+=list(hyperbounds[item])
                
        elif item == 'ub_noise':
            if isinstance(hyperbounds['ub_noise'], float):
                if hyperbounds['ub_noise']!=0:
                    upperbounds.append(np.log(hyperbounds[item]))
                else:
                    upperbounds.append(0)
            else :
                upperbounds+=list(np.log(hyperbounds[item]))
        elif item =='ub_theta':
            upperbounds+=list(np.log(hyperbounds[item]))
   
    lowerbounds = np.matrix(np.array(lowerbounds)).ravel()
    upperbounds = np.matrix(np.array(upperbounds)).ravel()
    nhyper = np.size(upperbounds)
    if verbose == 1:
        print('> estimating hyperparameters')
        print('  > number of hyperparameters = '+ str(nhyper))
        print('  > lower transformed hyp   = '+ str(lowerbounds)) 
        print('  > upper transformed hyp   = '+ str(upperbounds))
     
    def goalfunctionlocal(x0): #x0 = transfhyper
        out = goalfunction(x0,x,yn,ebian,estdn)
        out = out[0,0]
        return out
    
    bounds=scipy.optimize.Bounds(lowerbounds.T, upperbounds.T, keep_feasible=False)
    opt = scipy.optimize.differential_evolution(goalfunctionlocal, bounds)
    transfhyper = np.matrix(opt.x)
    nbiasproxy = ebian.shape[1]
    nStdProxy = estdn.shape[1]
    transfhyper = np.matrix(transfhyper)
    nhyper = transfhyper.shape[1]
    if nbiasproxy == 0:
        hyper = np.matrix(np.exp(transfhyper))
        ynb = yn.copy()
    else:
        biashyper = np.matrix(transfhyper[:,0:nbiasproxy])
        hyper = np.matrix(np.exp(transfhyper[:,nbiasproxy:nhyper]))
        ynb = yn - ebian*biashyper.transpose()
    hypernoise = hyper[:,0:nStdProxy]
    hypertheta = hyper[:,nStdProxy:]
    if verbose == 1:
        print('  > estimated transformed hyp = '+str(transfhyper))
        print('    > estimated eBias factors in celcius= '+str(biashyper))
        print('    > estimated eStd factors in celcius = '+str(hypernoise))
        print('    > estimated thetas in degrees = '+str(hypertheta))
    # compute initial state
    if verbose == 1:
        print('> computing initial state')
    A = buildA(x,estdn,hypernoise, hypertheta)
    #conditionnumber(A)
    #ploteigvec(A)
    y0n = cholesky(A,ynb)
    # output
    if verbose == 1:
        print('> returning krigingLE model')
    model = {'x': x, 'estdn': estdn, 'mu': mu, 'st': st, 
             'biashyper0': biashyper, 'hypertheta0': hypertheta, 'hypernoise0' : hypernoise,
             'y0n': y0n}
    return model
    

def predictMean(model,xi):      
    # predict value
    b = corrmatrix(xi,model['x'],model['hypertheta0'])
    yin = b*model['y0n']
                
    # denormalise prediction
    yi = model['mu'] + model['st']*yin
    # output    
    return yi

    
def predictStd(model,xi):
    
    #inflationFactor = len(model['y0n'])/(len(model['y0n'])-len(model['hyper0'])-len(model['biashyper0']))
    
    # predict variance
    b = corrmatrix(xi,model['x'],model['hypertheta0'])
    A = buildA(model['x'],model['estdn'],model['hypernoise0'],model['hypertheta0'])
    #varyin = inflationFactor * ( 1. - np.diag(np.dot(b,np.linalg.solve(A,b.T))) )
    varyin = 1. - np.diag(np.dot(b,cholesky(A,b.T)))
    varyin = np.matrix(varyin).T
    uin = np.sqrt(varyin)
    # denormalise prediction
    ui = model['st']*uin
    
    # output    
    return ui
    
def corrmatrix(x1,x2,hyper):
    ndim = x1.shape[1]
    nhyper = hyper.shape[1]
    theta = hyper[:,(nhyper-ndim):(nhyper)]
    D = np.zeros((np.shape(x1)[0],np.shape(x2)[0]))
    for k in range(ndim):
        X2,X1 = np.meshgrid(np.squeeze(np.asarray(x2[:,k])),
                            np.squeeze(np.asarray(x1[:,k])))       
        H = X2-X1
        if ndim==1:
            D += -0.5 * H**2 * np.squeeze(np.asarray(theta))**-2
        else:
            D += -0.5 * H**2 * np.squeeze(np.asarray(theta))[k]**-2
            
    C = np.exp(D)
    return C
    
def buildA(x,estdn,hypernoise, hypertheta):
    nestd = estdn.shape[1]
    P = corrmatrix(x,x,hypertheta) 
    R = np.zeros((np.shape(x)[0],np.shape(x)[0]))
    for k in range(np.shape(x)[0]):
        for n in range(nestd):
            #R[k,k] = R[k,k] + hypernoise[:,n]*estdn[k,n]**2
            R[k,k] = R[k,k] + (max(hypernoise[:,n]*estdn[k,n],regulizer))**2
    A = P + R
    return A
    
def goalfunction(transfhyper,x,yn,ebian,estdn):
    nbiasproxy = ebian.shape[1]
    nStdProxy = estdn.shape[1]
    transfhyper = np.matrix(transfhyper)
    nhyper = transfhyper.shape[1]
    if nbiasproxy == 0:
        hyper = np.matrix(np.exp(transfhyper))
        ynb = yn.copy()
    else:
        biashyper = np.matrix(transfhyper[:,0:nbiasproxy])
        hyper = np.matrix(np.exp(transfhyper[:,nbiasproxy:nhyper]))
        ynb = yn - ebian*biashyper.transpose()
    hypernoise = hyper[:,0:nStdProxy]
    hypertheta = hyper[:,nStdProxy:]
    A = buildA(x,estdn,hypernoise, hypertheta)
    goal = np.sum(np.log(np.linalg.eigvals(A))) + np.dot(ynb.T,cholesky(A,ynb))   
    goal = np.real(goal) # why is this necessary?
    
    return goal


def cholesky(A,x):
    L = np.linalg.cholesky(A)  
    a = np.linalg.solve(L,x)
    b = np.linalg.solve(L.transpose(), a)
    return b


    