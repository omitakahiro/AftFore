import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys,datetime,time
from multiprocessing import Pool
from copy import deepcopy

##################################
##MCMC
##################################
def MCMC_DetermieStepSize(F_LG,para_ini,Data,cdt,stg,n_core,prior=[],opt=[]):
    
    step_size_list = np.array([0.06,0.08,0.1,0.12,0.15,0.2,0.25,0.3,0.4,0.5])
    m = len(step_size_list)
    
    p = Pool(n_core)
    rslt = []
    for i in range(m):
        stg_tmp = deepcopy(stg)
        stg_tmp['step_size'] = step_size_list[i]
        rslt.append( p.apply_async(MCMC,args=[F_LG,para_ini,Data,cdt,stg_tmp,200,prior,['print']]) )
    p.close()
    p.join()
    rslt = [ rslt[i].get() for i in range(m) ]
    
    step_size    = [ rslt[i][2] for i in range(m) ]
    r_accept     = [ rslt[i][3] for i in range(m) ]
    elapsed_time = [ rslt[i][4] for i in range(m) ]
    dtl = pd.DataFrame(np.vstack([step_size,r_accept,elapsed_time]).T,columns=['step_size','r_accept','elapsed_time'])
   
    opt_step_size = dtl.iloc[ np.argmin(np.fabs(dtl['r_accept'].values-0.5)) ]['step_size']
    
    return [opt_step_size,dtl]

def MCMC_prl(F_LG,para_ini,Data,cdt,stg,Num,n_core,prior=[],opt=[]):
    
    print( "MCMC" )
    
    ##determine step size
    [opt_step_size,dtl1] = MCMC_DetermieStepSize(F_LG,para_ini,Data,cdt,stg,n_core,prior,opt)
    stg['step_size'] = opt_step_size
    
    print( "estimated processing time %.2f minutes" % (dtl1['elapsed_time'].mean()*Num/200.0/60.0) )
    
    p = Pool(n_core)
    rslt = [ p.apply_async(MCMC,args=[F_LG,para_ini,Data,cdt,stg,Num,prior,opt])   for i in range(n_core)   ]
    p.close()
    p.join()
    rslt = [ rslt[i].get() for i in range(n_core) ]
    
    para_mcmc     = pd.concat([rslt[i][0].iloc[0::10] for i in range(n_core) ],ignore_index=True)
    L_mcmc        = np.array([ rslt[i][1][0::10]      for i in range(n_core) ]).flatten()
    step_size     = np.array([ rslt[i][2]             for i in range(n_core) ])
    r_accept      = np.array([ rslt[i][3]             for i in range(n_core) ])
    elapsed_time  = np.array([ rslt[i][4]             for i in range(n_core) ])
    dtl2 = pd.DataFrame(np.vstack([step_size,r_accept,elapsed_time]).T,columns=['step_size','r_accept','elapsed_time'])
    
    dtl_mcmc = {'step_size':opt_step_size,'dtl1':dtl1,'dtl2':dtl2}
    
    return [para_mcmc,L_mcmc,dtl_mcmc]
    
    
def MCMC(F_LG,para_ini,Data,cdt,stg,Num,prior=[],opt=[]):
    
    #random number seed
    seed = datetime.datetime.now().microsecond *datetime.datetime.now().microsecond % 4294967295
    np.random.seed(seed)
    
    para_list  = stg['para_list']
    m          = stg['para_length']
    para_exp   = stg['para_exp']
    step_size  = stg['step_size']
    
    ##prior format transform
    if len(prior)>0:
        prior = pd.DataFrame(prior,columns=['name','type','mu','sigma'])
    
    #prepare
    para_mcmc = pd.DataFrame(index=np.arange(Num),columns=stg['para_list'],dtype='f8')
    L_mcmc    = np.zeros(Num)
    
    #initial value
    para1 = para_ini.copy()
    para_mcmc.iloc[0] = para1
    [L1,_] = Penalized_LG(F_LG,para1,Data,cdt,prior,only_L=True)
    L_mcmc[0] = L1
    
    #exponential parameter check
    para_ord = np.setdiff1d(para_list,para_exp)
    
    #step
    step_MCMC = stg['ste'].copy()
    step_MCMC[para_exp] =  np.minimum( np.log( 1.0 + step_MCMC[para_exp]/para1[para_exp] ) ,0.4)
    
    i = 1
    j = 0
    k = 0
    t_start = time.time()
    
    while 1:
        
        para2 = para1.copy()
        para2[para_ord] = para1[para_ord] +         step_size*np.random.randn(len(para_ord))*step_MCMC[para_ord]
        para2[para_exp] = para1[para_exp] * np.exp( step_size*np.random.randn(len(para_exp))*step_MCMC[para_exp] )
        
        [L2,_] = Penalized_LG(F_LG,para2,Data,cdt,prior,only_L=True)
        
        if L1<L2 or np.random.rand() < np.exp(L2-L1): #accept
            
            j += 1
            k += 1
            para1 = para2
            L1 = L2
            
            para_mcmc.iloc[i] = para1
            L_mcmc[i] = L1
            
        else:
            
            para_mcmc.iloc[i] = para_mcmc.iloc[i-1]
            L_mcmc[i] = L_mcmc[i-1]
        
        if 'print' in opt and np.mod(i,1000) == 0:
            print(i)
        
        #adjust the step width
        if np.mod(i,500) == 0:
            
            if k<250:
                step_size *= 0.95
            else:
                step_size *= 1.05
            
            k = 0
        
        i += 1
        
        if i == Num:
            break
    
    
    r_accept = 1.0*j/Num
    elapsed_time = time.time() - t_start
    
    return [para_mcmc,L_mcmc,step_size,r_accept,elapsed_time]

##################################
##Quasi-Newton
##################################
def Quasi_Newton(F_LG,para_ini,Data,cdt,stg,prior=[],opt=[]):
    
    ##parameter setting
    para_list  = stg['para_list']
    m          = stg['para_length']
    step_Q     = stg['para_step_Q'][para_list].values
    para_exp   = stg['para_exp']
    
    ##initial value
    para = para_ini
    
    ##prior format transform
    if len(prior)>0:
        prior = pd.DataFrame(prior,columns=['name','type','mu','sigma'])
    
    ##fix check
    if len(prior)>0:
        para_fix   = prior[ prior['type']=='f' ]['name'].values.astype('S')
        para_value = prior[ prior['type']=='f' ]['mu'].values
        para[para_fix] = para_value

    ##exponential parameter check
    para_ord = np.setdiff1d(para_list,para_exp)

    #calculate Likelihood and Gradient for the initial state
    [L,G1] = Penalized_LG(F_LG,para,Data,cdt,prior)
    G1[para_exp] = G1[para_exp] * para[para_exp]
    G1 = G1[para_list].values
    
    #OPTION return likelihood
    if 'L' in opt:
        return [para,L,[]]
    
    ###main
    H = np.eye(m)
    
    while 1:
        
        #OPTION: print [para,L]
        if 'print' in opt:
            for para_name in para_list:
                print( "%s: %e" % (para_name,para[para_name]) )
            print( "L = %.3f, norm(G) = %e\n" % (L,np.linalg.norm(G1)) )
        
        #break rule
        if np.linalg.norm(G1) < 1e-4:
            break
        
        #calculate direction
        s = H.dot(G1)
        s = s/np.max([np.max(np.abs(s)/step_Q),1])
        
        #update parameter value
        s_series = pd.Series(s,index=para_list)
        para[para_ord] = para[para_ord]  + s_series[para_ord]
        para[para_exp] = para[para_exp]  * np.exp(s_series[para_exp])
        
        #calculate Likelihood and Gradient
        [L,G2] = Penalized_LG(F_LG,para,Data,cdt,prior)
        G2[para_exp] = G2[para_exp] * para[para_exp]
        G2 = G2[para_list].values
        
        #update hessian matrix
        y=G1-G2;
        y = y.reshape(m,1)
        s = s.reshape(m,1)
        
        if  y.T.dot(s) > 0:
            H = H + (y.T.dot(s)+y.T.dot(H).dot(y))*(s*s.T)/(y.T.dot(s))**2.0 - (H.dot(y)*s.T+(s*y.T).dot(H))/(y.T.dot(s))
        else:
            H = np.eye(m)
        
        #update Gradients
        G1 = G2
    
    ###OPTION: Estimation Error
    if 'ste' in opt:
        ste = Estimation_Error(F_LG,para,Data,cdt,stg,prior)
    else:
        ste = []
    
    ###OPTION: Check map solution
    if 'check' in opt:
            ste = Check_QN(F_LG,para,Data,cdt,stg,prior)
    
    return [para,L,ste]

#################################
def Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,d):
    
    para_list = stg['para_list']
    para_tmp = para.copy()
    para_tmp[para_name] =  para_tmp[para_name] + d
    
    [_,G] = Penalized_LG(F_LG,para_tmp,Data,cdt,prior)
    G = G[para_list].values
    
    return G

def Hessian_Numerical(F_LG,para,Data,cdt,stg,prior):
    
    para_list = stg['para_list']
    m         = stg['para_length']
    para_exp  = stg['para_exp']
    para_step_H = stg['para_step_H']
    
    d = para_step_H.copy()
    d[para_exp] = d[para_exp] * para[para_exp]
    
    H = np.zeros([m,m])
    
    for i in range(m):
        para_name = para_list[i]
        
        """
        G1 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,-1.0*d[para_name])
        G2 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name, 1.0*d[para_name])
        H[:,i] = (G2-G1)/d[para_name]/2.0
        """
        
        G1 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,-2.0*d[para_name])
        G2 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name,-1.0*d[para_name])
        G3 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name, 1.0*d[para_name])
        G4 = Gradient_dev(F_LG,para,Data,cdt,stg,prior,para_name, 2.0*d[para_name])
        H[:,i] = (G1-8.0*G2+8.0*G3-G4)/d[para_name]/12.0
    
    return H

def Estimation_Error(F_LG,para,Data,cdt,stg,prior):
    
    para_list = stg['para_list']
    m         = stg['para_length']
    
    H = Hessian_Numerical(F_LG,para,Data,cdt,stg,prior)
    
    if len(prior)>0:
        para_fix = prior[prior['type']=='f']['name'].values.astype('S')
        index_ord = np.array([ not (para_name in para_fix) for para_name in para_list ])
    else:
        index_ord = np.repeat(True,m)
    
    C = np.zeros([m,m])
    C[np.ix_(index_ord,index_ord)] = np.linalg.inv(-H[np.ix_(index_ord,index_ord)])
    ste = pd.Series(np.sqrt(C.diagonal()),para_list)
    
    return ste

#################################
def Check_QN(F_LG,para,Data,cdt,stg,prior):
    
    para_list = stg['para_list']
    ste = Estimation_Error(F_LG,para,Data,cdt,stg,prior)
    a = np.arange(-1.0,1.1,0.2); L = np.zeros_like(a);
    
    for i,para_name in enumerate(para_list):
        
        plt.figure()
        for j in range(len(a)):
            para_tmp = para.copy()
            para_tmp[para_name] = para_tmp[para_name] + a[j]*ste[para_name]
            L[j] = Penalized_LG(F_LG,para_tmp,Data,cdt,prior,only_L=True)[0]
        
        plt.plot(para[para_name]+a*ste[para_name],L,'ko')
        plt.plot(para[para_name],L[5],'ro')
    
    return ste

#################################
##penalized likelihood
#################################
def Penalized_LG(F_LG,para,Data,cdt,prior,only_L=False):
    
    [L,G] = F_LG(para,Data,cdt,only_L=only_L)
    
    if len(prior)>0:
        
        ##Likelihood
        for i in range(len(prior)):
            [para_name,prior_type,mu,sigma] = prior.iloc[i][['name','type','mu','sigma']].values
            x = para[para_name]
            
            if prior_type == 'n': #prior: normal distribution
                L            = L - np.log(2.0*np.pi*sigma**2.0)/2.0 - (x-mu)**2.0/2.0/sigma**2.0
            elif prior_type ==  'ln': #prior: log-normal distribution
                L            = L - np.log(2.0*np.pi*sigma**2.0)/2.0 - np.log(x) - (np.log(x)-mu)**2.0/2.0/sigma**2.0
        
        ##Gradient
        if only_L == False:
            
            #prior
            for i in range(len(prior)):
                [para_name,prior_type,mu,sigma] = prior.iloc[i][['name','type','mu','sigma']].values
                x = para[para_name]
                
                if prior_type == 'n': #prior: normal distribution
                    G[para_name] = G[para_name] - (x-mu)/sigma**2.0
                elif prior_type ==  'ln': #prior: log-normal distribution
                    G[para_name] = G[para_name] - 1.0/x - (np.log(x)-mu)/sigma**2.0/x
            
            #fix
            para_fix = prior[prior['type']=='f']['name'].values.astype('S')
            G[para_fix] = 0
    
    return [L,G]