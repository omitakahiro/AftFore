import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from scipy.special import gammaln
from scipy.stats import norm,poisson

import sys,time,copy,pickle,datetime,subprocess

from StatTool import Quasi_Newton,MCMC_prl
from catalog import load_Data,parse_str
import gr

"""
model description:
lambda(t,M) = k*(t+c)**(-p) * beta*exp(-beta(M-mag_ref)) * Phi(M|mu(t),sigma)
"""
#####################################################
##generic values
#####################################################
def prior_generic():
    prior =[]
    prior.append(['beta','n',0.85*np.log(10),0.15*np.log(10)])
    prior.append(['p','n',1.05,0.13])
    prior.append(['c','ln',-4.02,1.42])
    prior.append(['sigma','ln',np.log(0.2),1.0])
    return prior

def para_generic():
    para = pd.Series({'beta':0.85*np.log(10),'k':np.exp(-4.86),'p':1.05,'c':np.exp(-4.02)})
    return para

#####################################################
##MCMC
#####################################################
def MCMC_OU_GRDR(Data,t,mag_ref,Num,n_core,prior=[],opt=[]):
    
    Data = Data.search({'t':t})
    
    #custom setting
    param_map = Estimate_OU_GRDR(Data,t,mag_ref,prior,['ste'])
    para_ini = param_map['para']
    ste      = param_map['ste']
    mu0      = param_map['mu0']
    cdt = {'t':t, 'mag_ref':mag_ref, 'mu0':mu0 }
    
    #setting
    para_list   = np.array(['beta','mu1','sigma','k','p','c'])
    para_length = 6
    para_exp    = np.array(['sigma','k','c'])
    stg = {'para_list':para_list,'para_length':para_length,'para_exp':para_exp,'ste':ste,'step_size':0.0}
    
    [para_mcmc,L_mcmc,dtl_mcmc] = MCMC_prl(LG_OU_GRDR,para_ini,Data,cdt,stg,Num,n_core,prior,opt)
    
    return {'para_mcmc':para_mcmc,'L_mcmc':L_mcmc,'para':para_mcmc.iloc[0],'L':L_mcmc[0],'t':t,'mag_ref':mag_ref,'mu0':mu0,'ste':ste,'prior':prior,'dtl_mcmc':dtl_mcmc}

#####################################################
##Estimation
#####################################################   
def Estimate_OU_GRDR(Data,t,mag_ref,prior=[],opt=[]):
    
    Data = Data.search({'t':t})
    
    #setting
    para_list   = ['beta','mu1','sigma','k','p','c']
    para_length = 6
    para_exp    = ['sigma','k','c']
    para_ini    = pd.Series({'beta':1.8,  'mu1':0.0,  'sigma':0.2,  'k':100.0, 'p':1.11, 'c': 1e-2})
    para_step_Q = pd.Series({'beta':0.2,  'mu1':0.1,  'sigma':0.2,  'k':1.0,   'p':0.1,  'c': 0.3})
    para_step_H = pd.Series({'beta':0.01, 'mu1':0.01, 'sigma':0.05, 'k':0.05,  'p':0.01, 'c': 0.05})
    
    stg = {'para_list':para_list,'para_length':para_length,'para_ini':para_ini,'para_step_Q':para_step_Q,'para_exp':para_exp,'para_step_H':para_step_H}
    
    #condition
    cdt = {'t':t, 'mag_ref':mag_ref, 'mu0':[]}
    
    #initial value
    prior_gr = [ col for col in prior if col[0] in ['beta','sigma']]
    param_gr = gr.Estimate_GRDR_SSM(Data,t,prior=prior_gr)
    cdt['mu0'] = param_gr['mu']

    para = para_ini
    para['beta'] = param_gr['para']['beta']
    para['sigma'] = param_gr['para']['sigma']
    n = len(Data.search({'t':t}))
    para['k'] = para['k'] * n / Int_OU_GRDR(para,Data,cdt,only_L=True)[0]
    
    #Quasi Newton
    [para,L,ste] = Quasi_Newton(LG_OU_GRDR,para,Data,cdt,stg,prior,opt)
    
    return {'para':para,'L':L,'t':t,'mag_ref':mag_ref,'mu0':param_gr['mu'],'ste':ste,'prior':prior}


def LG_OU_GRDR(para,Data,cdt,only_L=False):
    
    [Sum,d_Sum] = Sum_OU_GRDR(para,Data,cdt,only_L)
    [Int,d_Int] = Int_OU_GRDR(para,Data,cdt,only_L)

    L = Sum - Int
    G = d_Sum - d_Int
    
    return [L,G]

def Sum_OU_GRDR(para,Data,cdt,only_L=False):
    
    t = cdt['t']; mag_ref = cdt['mag_ref']; mu0 = cdt['mu0'];
    
    [sum_OU,   d_sum_OU  ] = Sum_OU(para,Data,t,only_L)
    [sum_GRDR, d_sum_GRDR] = Sum_GRDR(para,Data,cdt,only_L)
    
    #Sum
    Sum = sum_OU + sum_GRDR
    
    #d_Sum
    d_Sum = pd.Series()
    if only_L == False:
        d_Sum['beta']  = d_sum_GRDR['beta']
        d_Sum['mu1']   = d_sum_GRDR['mu1']
        d_Sum['sigma'] = d_sum_GRDR['sigma']
        d_Sum['k']     = d_sum_OU['k']
        d_Sum['p']     = d_sum_OU['p']
        d_Sum['c']     = d_sum_OU['c']
    
    return [Sum,d_Sum]

def Int_OU_GRDR(para,Data,cdt,only_L=False):
    
    t = cdt['t']; mag_ref = cdt['mag_ref']; mu0 = cdt['mu0'];
    #T= Data.search({'t':t})['T'].values
    #T = Data['T'].values
    T= Data[ (t['st']<Data['T']) & (Data['T']<t['en']) ]['T'].values
    st_bin = np.append(t['st'],T); en_bin = np.append(T,t['en']); t_bin = {'st':st_bin,'en':en_bin};
    
    [int_GRDR, d_int_GRDR] = Int_GRDR(para,cdt,only_L)
    [int_OU,   d_int_OU  ] = Int_OU(para,t_bin,only_L)
    
    int_GRDR = np.append(int_GRDR,int_GRDR[-1])
    for para_ix in d_int_GRDR.keys():
        d_int_GRDR[para_ix] = np.append(d_int_GRDR[para_ix],d_int_GRDR[para_ix][-1])
    
    #Int
    Int = ( int_OU * int_GRDR ).sum()
    
    #d_Int
    d_Int = pd.Series()
    if only_L == False:
        d_Int['beta']  = ( int_OU        * d_int_GRDR['beta']  ).sum()
        d_Int['mu1']   = ( int_OU        * d_int_GRDR['mu1']   ).sum()
        d_Int['sigma'] = ( int_OU        * d_int_GRDR['sigma'] ).sum()
        d_Int['k']     = ( d_int_OU['k'] * int_GRDR            ).sum()
        d_Int['p']     = ( d_int_OU['p'] * int_GRDR            ).sum()
        d_Int['c']     = ( d_int_OU['c'] * int_GRDR            ).sum()
    
    return [Int,d_Int]

#####################################################
def Sum_GRDR(para,Data,cdt,only_L=False):
    
    t = cdt['t']; mag_ref = cdt['mag_ref']; mu0 = cdt['mu0'];
    beta = para['beta']; mu = mu0 + para['mu1']; sigma = para['sigma'];
    #Mag = Data.search({'t':t})['Mag'].values
    #Mag = Data['Mag'].values
    Mag = Data[ (t['st']<Data['T']) & (Data['T']<t['en']) ]['Mag'].values
    n_Mag = (Mag-mu)/sigma
        
    Sum = ( np.log(beta) - beta*(Mag-mag_ref) + np.log( norm.cdf(n_Mag) ) ).sum()
    
    d_Sum = pd.Series()
    if only_L == False:
        d_Sum['beta']  = ( 1.0/beta - (Mag-mag_ref)                                        ).sum()
        d_Sum['mu1']   = (                          +  (-1.0/  sigma)*gr.d_Lnormcdf(n_Mag) ).sum()
        d_Sum['sigma'] = (                          +  (-n_Mag/sigma)*gr.d_Lnormcdf(n_Mag) ).sum()
    
    return [Sum,d_Sum]

def Int_GRDR(para,cdt,only_L=False):
    
    t = cdt['t']; mag_ref = cdt['mag_ref']; mu0 = cdt['mu0'];
    beta = para['beta']; mu = mu0 + para['mu1']; sigma = para['sigma'];
    
    Int = np.exp( beta*(mag_ref-mu) + (beta*sigma)**2.0/2.0 )
    
    d_Int = pd.Series()
    if only_L == False:
        d_Int['beta']  = Int * ( (mag_ref-mu) + beta*sigma**2.0 )
        d_Int['mu1']   = Int * ( -beta                          )
        d_Int['sigma'] = Int * (                beta**2.0*sigma )
    
    return [Int,d_Int]
    
#####################################################
def Sum_OU(para,Data,t,only_L=False):
    
    k = para['k']; p = para['p']; c = para['c'];
    #T= Data.search({'t':t})['T'].values
    #T = Data['T'].values
    T= Data[ (t['st']<Data['T']) & (Data['T']<t['en']) ]['T'].values
    n = len(T)
    
    #Sum
    Sum = ( np.log(k) - p*np.log(T+c) ).sum()
    
    #d_Sum
    d_Sum = pd.Series()
    if only_L == False:
        d_Sum['k'] = n/k
        d_Sum['p'] = ( -np.log(T+c) ).sum()
        d_Sum['c'] = ( -p/(T+c)     ).sum()
    
    return [Sum,d_Sum]

def Int_OU(para,t,only_L=False):
    
    k = para['k']; p = para['p']; c = para['c'];
    st = t['st']; en = t['en'];
    
    f_st = (st+c)**(-p+1.0)/(-p+1.0); f_en = (en+c)**(-p+1.0)/(-p+1.0);
    
    #Int
    Int = k * ( f_en -f_st )
    
    #d_Int
    d_Int = {}
    if only_L == False:
        d_Int['k'] = (f_en-f_st)
        d_Int['p'] = -k * ( np.log(en+c)*f_en - np.log(st+c)*f_st ) + k * (f_en-f_st) /(-p+1.0)
        d_Int['c'] = k * ( (en+c)**(-p) - (st+c)**(-p) )
    
    return [Int,d_Int]


"""
#####################################################
##Score
#####################################################   
def score_error(para1,mag_ref1,para2,mag_ref2,Data,mag_min,t):
    
    mag_bin = np.arange(mag_min,10.051,0.1)
    c_obs = calc_c_obs(Data,t,mag_bin)
    c_exp1 = calc_c_exp(para1,mag_ref1,t,mag_bin)
    c_exp2 = calc_c_exp(para2,mag_ref2,t,mag_bin)
    
    L_dif = np.zeros(1000)
    for i in np.arange(1000):
        
        if i==0:
            c_syn = c_obs
        else:
            c_syn = np.random.poisson(c_obs)
        
        L_dif[i] = score_poisson(c_exp1,c_syn) - score_poisson(c_exp2,c_syn)
    
    IG = L_dif[0]
    IG_var = np.var(L_dif)
    n = c_obs.sum()
    return [IG,IG_var,n]

def calc_c_obs(Data,t,mag_bin):
    T = Data[ (t['st']<Data['T']) & (Data['T']<t['en']) ]['Mag'].values
    c_obs = np.histogram(T,mag_bin)[0].reshape(1,-1)
    return c_obs

def calc_c_exp(para,mag_ref,t,mag_bin):
    
    if type(para) == type(pd.DataFrame()):
        k = para['k'].values.reshape(-1,1)
        p = para['p'].values.reshape(-1,1)
        c = para['c'].values.reshape(-1,1)
        beta = para['beta'].values.reshape(-1,1)
    elif type(para) == type(pd.Series()):
        k = para['k']
        p = para['p']
        c = para['c']
        beta = para['beta']
    
    mag_bin_l = mag_bin[:-1].reshape(1,-1)
    mag_bin_r = mag_bin[1:].reshape(1,-1)
    
    c_OU = np.exp(beta*(mag_ref-mag_bin[0])) * k*( (t['en']+c)**(-p+1) - (t['st']+c)**(-p+1) )/(-p+1)
    c_GR = np.exp(-beta*(mag_bin_l-mag_bin[0])) - np.exp(-beta*(mag_bin_r-mag_bin[0]))
    c_exp = c_OU*c_GR
    
    return c_exp
   
def score_poisson(c_exp,c_obs):
    L = ( c_obs*np.log(c_exp) - c_exp - gammaln(c_obs+1) ).sum(axis=1)
    ML = L[0] + np.log(np.mean(np.exp(L-L[0])))
    return ML
"""

#####################################################
##Predictive Distribution
#####################################################   
def calc_CumPredDist(para,mag_ref,t):
    
    [mag,c_cum_exp] = calc_c_cum_exp(para,mag_ref,t)
    m = len(mag)
    curve_map = np.zeros(m)
    curve_95  = np.zeros([2,m])
    
    for i,l in enumerate(c_cum_exp):
        curve_map[i] = l[0]
        curve_95[:,i] = calc_95range(l)
    
    pred = pd.DataFrame({'mag':mag,'expected_num':curve_map,'lower_bound':curve_95[0],'upper_bound':curve_95[1]})
    
    return pred

def calc_c_cum_exp(para,mag_ref,t):
    
    if type(para) == type(pd.DataFrame()):
        k    = para['k'   ].values.reshape(1,-1)
        p    = para['p'   ].values.reshape(1,-1)
        c    = para['c'   ].values.reshape(1,-1)
        beta = para['beta'].values.reshape(1,-1)
    elif type(para) == type(pd.Series()):
        k    = para['k'   ]
        p    = para['p'   ]
        c    = para['c'   ]
        beta = para['beta']
    
    st = t['st']; en = t['en'];
    mag = np.arange(0.95,10.0,0.1).reshape(-1,1); mag0 = mag[0,0];
    
    c_OU = k * ( (en+c)**(-p+1.0) - (st+c)**(-p+1.0) ) / (-p+1.0)
    c_GR = np.exp(  beta*(mag_ref-mag) )
    c_cum_exp = c_GR * c_OU
    
    return [mag.flatten(),c_cum_exp]

def calc_95range(l):
    
    c_min = 0  if l.min()<36.0 else ( np.floor( l.min() - np.sqrt(l.min())*6.0 ) ).astype('i') 
    c_max = 72 if l.max()<36.0 else ( np.ceil(  l.max() + np.sqrt(l.max())*6.0 ) ).astype('i') 
    x = np.arange(c_min,c_max+1)
    y = np.zeros_like(x,dtype='f8')

    for l_i in l:
        y = y + poisson.pmf(x,l_i)

    y = y/len(l)
    y_cdf = y.cumsum()
    x1 = x[y_cdf>0.025][0]
    x2 = x[y_cdf>0.975][0]
    
    if y.sum() < 0.999:
        sys.exit('range error in calc_95range')
    
    return np.array([x1,x2])

#####################################################
##Graph
#####################################################
def Graph_dr(Data,param,xlim=None,ylim=None,newfig=True,pdf=False):
    
    [beta,mu1,sigma] = param['para'][['beta','mu1','sigma']].values
    t = param['t']
    mu = param['para']['mu1'] + param['mu0']
    [T,Mag] = Data.search({'t':t})[['T','Mag']].values.T
    ylim = [Mag.min()-1.0,Mag.max()+1.0] if ylim is None else ylim
    
    if newfig is True:
        plt.figure(figsize=(8.27,11.69))
        plt.clf()
        mpl.rc('font', size=12, family='Arial')
        mpl.rc('axes',linewidth=1,titlesize=12)
        mpl.rc('pdf',fonttype=42)
        mpl.rc('xtick.major',width=1)
        mpl.rc('xtick.minor',width=1)
        mpl.rc('ytick.major',width=1)
        mpl.rc('ytick.minor',width=1)

    plt.plot(T,Mag,'o',mfc='#cccccc',mec='#cccccc')
    plt.plot(T,mu,'r-',linewidth=2)
    plt.plot([1./8.,1./8.],ylim,'k--',dashes=(1,1),linewidth=0.5)
    plt.plot([1./4.,1./4.],ylim,'k--',dashes=(1,1),linewidth=0.5)
    plt.plot([1./2.,1./2.],ylim,'k--',dashes=(1,1),linewidth=0.5)
    plt.plot([1./1.,1./1.],ylim,'k--',dashes=(1,1),linewidth=0.5)

    plt.xscale('log')
    plt.xlabel('time after the main shock')
    plt.ylabel('magnitude')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('beta: %.2f, sigma: %.2f, t: [%.3f,%.3f]'%(beta,sigma,t['st'],t['en']))

    if newfig is True:
        plt.tight_layout(rect=[0.25,0.3,0.75,0.7])

    if pdf is not False:
        if pdf is True:
            plt.savefig('mu.pdf')
        else:
            plt.savefig(pdf)

def Graph_param(param,newfig=True,pdf=False):
    
    if newfig is True:
        fig = plt.figure(figsize=(6,6))
        mpl.rc('font', size=12, family='Arial')
        mpl.rc('axes',linewidth=1,titlesize=12)
        mpl.rc('pdf',fonttype=42)
        mpl.rc('xtick.major',width=1)
        mpl.rc('xtick.minor',width=1)
        mpl.rc('ytick.major',width=1)
        mpl.rc('ytick.minor',width=1)
    
    p = param['para_mcmc'].copy()
    p['log(k)'] = np.log(p['k'])
    p['log(c)'] = np.log(p['c'])
    p['log(sigma)'] = np.log(p['sigma'])
    
    para_list = ['log(k)','p','log(c)','beta']
    
    axs = pd.tools.plotting.scatter_matrix(p[para_list],alpha=1,figsize=(6,6),c=[0.7,0.7,0.7],linewidth=0,range_padding=0.5)

    for i,x in enumerate(para_list):
        for j,y in enumerate(para_list):
            if i != j:
                axs[i,j].plot(p[y][0],p[x][0],'ro')
    
    if pdf is not False:
        if pdf is True:
            plt.savefig('scatter_matrix.pdf')
        else:
            plt.savefig(pdf)

def Graph_pred(pred,param,t_test,xlim=None,ylim=None,newfig=True,pdf=False):
    
    k = param['para']['k']; p = param['para']['p']; c = param['para']['c']; beta = param['para']['beta']
    t1 = t_test['st']; t2 = t_test['en']
    m0 = param['mag_ref']
    
    ylim = [0.8,1000] if ylim is None else ylim
    xlim = [1.0,7.0] if xlim is None else xlim
    mag = pred['mag']
    x_e  = np.vstack([mag,mag+0.1]).T.flatten()
    y_e1 = np.repeat(pred['lower_bound'].values,2) 
    y_e2 = np.repeat(pred['upper_bound'].values,2) 
    
    y_e1[y_e1<1] = 0.8; y_e2[y_e2<1] = 0.8;
    
    if newfig is True:
        plt.figure(figsize=(8.27,11.69))
        plt.clf()
        mpl.rc('font', size=12, family='Arial')
        mpl.rc('axes',linewidth=1,titlesize=12)
        mpl.rc('pdf',fonttype=42)
        mpl.rc('xtick.major',width=1)
        mpl.rc('xtick.minor',width=1)
        mpl.rc('ytick.major',width=1)
        mpl.rc('ytick.minor',width=1)
    
    plt.plot(mag+0.05,pred['expected_num'],'r-',label='expected')
    plt.fill_between(x_e,y_e1,y_e2,facecolor='r',linewidth=0,alpha=0.2)
    
    if 'expected_num_generic' in pred.columns:
        plt.plot(mag+0.05,pred['expected_num_generic'],'g-',label='generic model')
    
    if 'c_obs1' in pred.columns:
        plt.plot(mag+0.05,pred['c_obs1'],'ko',label='Hi-net')
        
    if 'c_obs2' in pred.columns:
        plt.plot(mag,pred['c_obs2'],'ks',mfc='w',label='JMA')
        
    plt.xlabel('magnitude')
    plt.ylabel('cumulative number')
    plt.yscale('log')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='best',numpoints=1,fontsize=12)
    plt.title('k: %.2e, p: %.2f, c: %.2e, beta: %.2f, t_test:[%.3f,%.3f], M_0: %.1f'%(k,p,c,beta,t1,t2,m0))

    if newfig is True:
        plt.tight_layout(rect=[0.25,0.3,0.75,0.7])

    if pdf is not False:
        if pdf is True:
            plt.savefig('pred.pdf')
        else:
            plt.savefig(pdf)

#####################################################
## Tool for real-time analysis
#####################################################
class RT_Tool():
    
    def parse_str(self,s):
        return parse_str(s)
    
    def prepare_data(self,f_catalog,ms,alpha,xy,duration,prefix):
        
        Data = load_Data(f_catalog)
        Data.plot_map(ms=ms,aftzone=True,pdf='%s_hypo_check.pdf'%prefix)
        [Data_seq,xy] = Data.extract_aft(ms,alpha=alpha,xy=xy,duration=duration)
        Data_seq.plot_map(ms=ms,txm=xy,pdf='%s_hypo.pdf'%prefix)
        Data_seq.plot_mt_aft(pdf='%s_mt.pdf'%prefix)
        Data_seq.save_Data('%s_aft.dat'%prefix)
        
        dtl = {'ms':ms, 'alpha':alpha, 'Catalog':Data, 'Data':Data_seq,'xy':xy}
        pickle.dump(dtl,open('%s_data.pkl'%prefix,'wb'),protocol=2)
        
        return Data_seq
    
    def estimate_para(self,Data,t,mag_ref,prior=prior_generic(),opt=[]):
        
        param = MCMC_OU_GRDR(Data,t,mag_ref,1000,2,prior=prior,opt=opt)
        #param['para_mcmc'] = param['para_mcmc'].iloc[::5]
        #param['para_mcmc'].index = np.arange(1000)
        pickle.dump(param,open('AAA_param.pkl','wb'),protocol=2)
        
        pp = PdfPages('AAA_param.pdf')
        Graph_dr(Data,param,xlim=None,ylim=None,newfig=True)
        pp.savefig()
        Graph_param(param,newfig=True)
        pp.savefig()
        pp.close()
        
        self.predictive_dist(param,t,Data_test1=Data,prefix='AAA_check')
        self.predictive_dist(param,{'st':t['en'],'en':2.0*t['en']},prefix='AAA')
        
        return param
    
    def predictive_dist(self,param,t_fore,para_ref=para_generic(),Data_test1=None,Data_test2=None,prefix='AAA'):
        
        para_mcmc = param['para_mcmc']; mag_ref = param['mag_ref'];
        pred     = calc_CumPredDist(para_mcmc,mag_ref,t_fore)
        pred_ref = calc_CumPredDist(para_ref ,mag_ref,t_fore)
        mag = pred['mag']
        pred['expected_num_generic'] = pred_ref['expected_num']
        pred.t = t_fore
                
        if Data_test1 is not None:
            [_,c_obs1] = Data_test1.mfd_cum(txm={'t':t_fore},mag=mag)
            pred['c_obs1'] = c_obs1
            
        if Data_test2 is not None:
            [_,c_obs2] = Data_test2.mfd_cum(txm={'t':t_fore},mag=mag)
            pred['c_obs2'] = c_obs2

        pickle.dump(pred,open('%s_pred.pkl'%prefix,'wb'),protocol=2)

        Graph_pred(pred,param,t_fore,xlim=None,ylim=None,newfig=True,pdf='%s_pred.pdf'%prefix)
        
        return pred
        
        
####################################
##demo
####################################
def demo():
    Data = load_Data('Hyogo.txt')
    
    t = {'st':0.0,'en':1.0}
    mag_ref = Data['Mag'][0]
    
    prior =[]
    prior.append(['beta','n',0.85*np.log(10),0.15*np.log(10)])
    prior.append(['p','n',1.12,0.11])
    prior.append(['c','ln',-4.29,1.09])
    prior.append(['sigma','ln',np.log(0.2),1.0])
    
    opt = []

    param = Estimate_OU_GRDR(Data,t,mag_ref,prior=prior,opt=opt)
    print(param['para'][['beta','mu1','sigma','k','p','c']])
    print(param['L'])
    print(param['ste'])
    
    """
    mag_min = 1.45
    T = Data[Data['Mag']>mag_min]['T'].values
    plt.figure(110)
    plt.clf()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001,1000)
    bin_edge = 2.0**np.arange(-10,10)
    l_obs = np.histogram(T,bin_edge)[0]/(bin_edge[1:]-bin_edge[:-1])
    plt.plot(np.sqrt(bin_edge[1:]*bin_edge[:-1]),l_obs,'ko')
    t = 2.0**np.arange(-10,10,0.1)
    l_ou = np.exp(para['beta']*(mag_ref-mag_min)) * para['k']*(t+para['c'])**(-para['p'])
    plt.plot(t,l_ou,'r-')
    """


def demo_MCMC():
    Data = load_Data('Hyogo.txt')
    t = {'st':0.0,'en':1.0}
    mag_ref = Data['Mag'][0]

    prior =[]
    prior.append(['beta','n',0.85*np.log(10),0.15*np.log(10)])
    prior.append(['p','n',1.12,0.11])
    prior.append(['c','ln',-4.29,1.09])
    prior.append(['sigma','ln',np.log(0.2),1.0])
    
    opt = ['print']
    
    param = MCMC_OU_GRDR(Data,t,mag_ref,1000,2,prior,opt)
    return param

def demo_RT():
    ######################################################################### Setting
    ##main shock information
    ms_s = '2016-04-16 01:25:05.47 130.763 32.7545 12.45 7.3'
    alpha = 3.0
    
    #file information
    f_catalog = 'kuma.jma'
    
    #forecast information
    t_h = 3 #hours
    
    ######################################################################### Forecast
    RT = RT_Tool()
    t1 = t_h/24.0
    dir_output = '%dh' % t_h
    ms = RT.parse_str(ms_s)
    
    print(ms)
    print('\n')
    
    ####data preparation
    Data = RT.prepare_data(f_catalog,ms,alpha,None,t1*2.0,'AAA')
    
    ##estimate parameters
    param = RT.estimate_para(Data,{'st':0.0,'en':t1},ms['Mag'])
    
    ####create a new directory and move files to the directory
    subprocess.call('mkdir %s' % dir_output, shell=True)
    subprocess.call('mv AAA* %s'  % dir_output, shell=True)
    
    
    #predictive distribution - test
    pred = RT.predictive_dist(param,{'st':t1,'en':t1*2.0},Data_test1=Data,prefix='BBB')
    subprocess.call('mv BBB* %s'  % dir_output, shell=True)
    
    
####################################
##Test *never change the script
####################################
def TEST():
    Data = load_Data('Hyogo.txt')
    t = {'st':0.0,'en':1.0}
    mag_ref = Data['Mag'][0]
    
    prior =[]
    prior.append(['beta','n',0.85*np.log(10),0.15*np.log(10)])
    prior.append(['p','n',1.12,0.11])
    prior.append(['c','ln',-4.29,1.09])
    prior.append(['sigma','ln',np.log(0.2),1.0])
    
    opt = ['ste']
    
    param = Estimate_OU_GRDR(Data,t,mag_ref,prior=prior,opt=opt)
    print(param['para'][['beta','mu1','sigma','k','p','c']])
    print(param['L'])
    print(param['ste'])
    
    """OUTPUT
    beta     1.702350
    mu1     -0.032289
    sigma    0.237203
    k        0.020285
    p        1.057055
    c        0.016753
    dtype: float64
    4359.85141167
    beta     0.063909
    mu1      0.032397
    sigma    0.016502
    k        0.007286
    p        0.060326
    c        0.006400
    dtype: float64
    """
####################################
##Main
####################################
if __name__ == '__main__':
    pass
