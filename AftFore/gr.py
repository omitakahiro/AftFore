import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


from scipy.stats import norm 
from scipy import sparse
import scipy.sparse.linalg as spla

from .StatTool import Quasi_Newton

#####################################################
##GR and detection rate model with fixed parameters
#####################################################
def Estimate_GRDR(Data,t,prior=[],opt=[]):
    
    #setting
    para_list   = np.array(['beta','mu','sigma'])
    para_length = 3
    para_exp    = np.array(['sigma'])
    para_ini    = pd.Series({'beta':2.0,  'mu':2.0,  'sigma':0.2})
    para_step_Q = pd.Series({'beta':0.2,  'mu':0.2,  'sigma':0.2})
    para_step_H = pd.Series({'beta':0.01, 'mu':0.01, 'sigma':0.01})
    
    stg = {'para_list':para_list,'para_length':para_length,'para_ini':para_ini,'para_step_Q':para_step_Q,'para_exp':para_exp,'para_step_H':para_step_H}
    
    #condition
    cdt = {'t':t}
    
    #optimization
    [para,L,ste] = Quasi_Newton(LG_GRDR,para_ini,Data,cdt,stg,prior,opt)
    
    ##mag_top
    mag_top = mode_pdf_GRDR(para)
    
    if 'fig' in opt:
        Graph_GRDR(para,Data,t)
    
    return {'para':para,'L':L,'ste':ste,'mag_top':mag_top,'t':t}

def LG_GRDR(para,Data,cdt,only_L=False):
    
    t = cdt['t']
    Mag = Data.search({'t':t})['Mag'].values
    
    [L,G_GF] = LG_pdf_GRDR(para,Mag,only_L)
    
    G = pd.Series()
    if only_L == False:
        G['beta']  = G_GF['beta'].sum()
        G['mu']    = G_GF['mu'].sum()
        G['sigma'] = G_GF['sigma'].sum()
    
    return [L,G]

def Graph_GRDR(para,Data,t,xlim=None,newfig=True,pdf=False):
    
    Mag = Data.search({'t':t})['Mag'].values
    [x,c] = Data.mfd({'t':t})
    mag_top = mode_pdf_GRDR(para)
    
    if newfig is True:
        plt.figure(figsize=(8.27,11.69))
        mpl.rc('font', size=12, family='Arial')
        mpl.rc('axes',linewidth=1,titlesize=12)
        mpl.rc('pdf',fonttype=42)
        
    plt.plot(x,c,'ko')
    plt.plot(x,pdf_GRDR(para,x)*0.1*c.sum(),'r-',linewidth=2)
    plt.plot([mag_top,mag_top],[0.5,max(c)*2],'k:')
    
    plt.yscale('log')
    plt.xlabel('magnitude')
    plt.ylabel('count')
    plt.ylim([0.5,max(c)*2])
    plt.xlim([Mag.min()-1.0,Mag.max()+1.0] if xlim is None else xlim)
    
    if t is None:
        plt.title('beta: %.2f, mu: %.2f, sigma: %.2f' % (para['beta'],para['mu'],para['sigma']))
    else:
        plt.title('beta: %.2f, mu: %.2f, sigma: %.2f, T: [%.3f,%.3f]' % (para['beta'],para['mu'],para['sigma'],t['st'],t['en']))
    
    if newfig is True:
        plt.tight_layout(rect=[0.2,0.3,0.8,0.6])
    
    if pdf is not False:
        if pdf is True:
            plt.savefig('grdr.pdf')
        else:
            plt.savefig(pdf)
    
#####################################################
##GR and detection rate model with time-varying mu
#####################################################
def Estimate_GRDR_SSM(Data,t,prior=[],opt=[]):
    
    #setting
    para_list   = np.array(['beta','sigma','V'])
    para_length = 3
    para_exp    = np.array(['sigma','V'])
    para_ini    = pd.Series({'beta':2.0,  'sigma':0.2,  'V':1e-6})
    para_step_Q = pd.Series({'beta':0.2,  'sigma':0.2,  'V':0.2})
    para_step_H = pd.Series({'beta':0.01, 'sigma':0.01, 'V':0.01})
    
    stg = {'para_list':para_list,'para_length':para_length,'para_ini':para_ini,'para_step_Q':para_step_Q,'para_exp':para_exp,'para_step_H':para_step_H}
    
    #condition
    Mag = Data.search({'t':t})['Mag'].values
    mu = Mag.mean() * np.ones_like(Mag)
    [W,rank_W] = Weight_Matrix_2nd(len(Mag))
    
    cdt = {'t':t, 'mu':mu, 'W':W, 'rank_W':rank_W}
    
    #optimization
    [para,L,ste] = Quasi_Newton(LG_GRDR_SSM,para_ini,Data,cdt,stg,prior,opt)
    Estimate_mu(para,mu,Mag,cdt)
    
    #figure
    if 'fig' in opt:
        Graph_MT_mu(para,mu,Data,t)
        
    return {'para':para,'L':L,'ste':ste,'t':t,'mu':mu.copy()}

def LG_GRDR_SSM(para,Data,cdt,only_L=False):
    
    t = cdt['t']; mu = cdt['mu'];
    beta = para['beta']; sigma = para['sigma']; V = para['V'];
    Mag = Data.search({'t':t})['Mag'].values
    
    #Likelihood
    L = M_L(para,mu,Mag,cdt)
    
    #Gradient
    G = pd.Series()
    para_list = np.array(['beta','sigma','V'])
    epsilon = pd.Series({'beta':0.01,'sigma':sigma*0.01,'V':V*0.01})
    
    for para_name in para_list:
        
        para1 = para.copy();  para1[para_name] = para1[para_name] - 2.0*epsilon[para_name];  L1 = M_L(para1,mu.copy(),Mag,cdt);
        para2 = para.copy();  para2[para_name] = para2[para_name] - 1.0*epsilon[para_name];  L2 = M_L(para2,mu.copy(),Mag,cdt);
        para3 = para.copy();  para3[para_name] = para3[para_name] + 1.0*epsilon[para_name];  L3 = M_L(para3,mu.copy(),Mag,cdt);
        para4 = para.copy();  para4[para_name] = para4[para_name] + 2.0*epsilon[para_name];  L4 = M_L(para4,mu.copy(),Mag,cdt);
        
        G[para_name] = ( L1 - 8.0*L2 + 8.0*L3 - L4 )/12.0/epsilon[para_name]
        
        """
        para1 = para.copy(); para1[para_name] = para1[para_name] - 1.0*epsilon[para_name]; L1  = M_L(para1,mu.copy(),Mag,cdt);
        para2 = para.copy(); para2[para_name] = para2[para_name] + 1.0*epsilon[para_name]; L2  = M_L(para2,mu.copy(),Mag,cdt);
        
        G[para_name] = ( L2 - L1 )/2.0/epsilon[para_name]
        """
    
    return [L,G]

def M_L(para,mu,Mag,cdt):
    
    n = len(Mag)
    V = para['V']
    
    [L,G,H] = Estimate_mu(para,mu,Mag,cdt)
    
    LU = spla.splu(-H)
    log_det = np.log(np.abs(LU.U.diagonal())).sum()
    
    ML = L + np.log(2.0*np.pi)*n/2.0 - log_det/2.0
    ML = ML - np.exp(-(V-5e-8)/1e-8)
    
    return ML
    
def Estimate_mu(para,mu,Mag,cdt):
    
    while 1:
        [L,G,H]=LGH(para,mu,Mag,cdt)
        
        if np.linalg.norm(G)<1e-5:
            break
        
        d = -spla.spsolve(H,G)
        
        d_max = np.abs(d).max()
        if d_max > 1.0:
            d = d/d_max 
        
        mu[:] = mu + d
    
    return [L,G,H]
    
def LGH(para,mu,Mag,cdt):
    
    beta = para['beta']; sigma = para['sigma']; V = para['V'];
    W = cdt['W']; rank_W = cdt['rank_W'];
    n = len(Mag)

    ##LGH
    n_Mag=(Mag-mu)/sigma    
    L = ( np.log(beta) -beta*(Mag-mu) -(beta*sigma)**2.0/2.0 +np.log(norm.cdf(n_Mag)) ).sum() - rank_W*np.log(2*np.pi*V)/2.0 - mu.dot(W.dot(mu))/2.0/V 
    G = ( beta + d_Lnormcdf(n_Mag)*(-1.0/sigma) ) - W.dot(mu)/V
    H = sparse.csc_matrix(sparse.spdiags(d2_Lnormcdf(n_Mag)*(1.0/sigma**2.0),0,n,n)) - W/V
    
    return [L,G,H]

def Weight_Matrix_2nd(n):
    
    d0 = np.hstack(([1,5],6*np.ones(n-4),[5,1]))
    d1 = np.hstack((-2,-4*np.ones(n-3),-2))
    d2 = np.ones(n-2)
    
    data = np.array([d2,d1,d0,d1,d2])
    diags = np.arange(-2,3)
    W = sparse.diags(data,diags,shape=(n,n),format='csc')
    
    rank_W = n-2
    
    return [W,rank_W]

def Graph_MT_mu(para,mu,Data,t,xlim=None,ylim=None,newfig=True,pdf=False):
    
    [T,Mag] = Data.search({'t':t})[['T','Mag']].values.T
    ylim = [Mag.min()-1.0,Mag.max()+1.0] if ylim is None else ylim
    
    if newfig is True:
        plt.figure(figsize=(8.27,11.69))
        mpl.rc('font', size=12, family='Arial')
        mpl.rc('axes',linewidth=1,titlesize=12)
        mpl.rc('pdf',fonttype=42)
    
    plt.plot(T,Mag,'o',mfc='#cccccc',mec='#cccccc')
    plt.plot(T,mu,'r-',linewidth=2)
    plt.plot([t['en'],t['en']],ylim,'k--',dashes=(1,1),linewidth=0.5)
    
    plt.xscale('log')
    plt.xlabel('time after the main shock')
    plt.ylabel('magnitude')
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    if newfig is True:
        plt.tight_layout(rect=[0.2,0.3,0.8,0.6])
    
    if pdf is not False:
        if pdf is True:
            plt.savefig('grdr.pdf')
        else:
            plt.savefig(pdf)
    
#####################################################
##Basic fucntions
#####################################################
##probability density function
def pdf_GRDR(para,x):
    beta = para['beta'];  mu = para['mu'];  sigma = para['sigma'];
    return beta * np.exp(-beta*(x-mu)-(beta*sigma)**2.0/2.0) * norm.cdf((x-mu)/sigma)

def mode_pdf_GRDR(para):
    x = np.arange(-2.0,10.01,0.1)
    y = pdf_GRDR(para,x)
    return x[y.argmax()]

##log-likelihood function and its gradient
def LG_pdf_GRDR(para,Mag,only_L=False):
    
    beta = para['beta']; mu = para['mu']; sigma = para['sigma'];
    n_Mag = (Mag-mu)/sigma
    
    #Likelihood
    L = ( np.log(beta) - beta*(Mag-mu) - (beta*sigma)**2.0/2.0 + np.log( norm.cdf(n_Mag) ) ).sum()
    
    #Gradient
    G = {}
    if only_L == False:
        G['beta']  = ( 1.0/beta - (Mag-mu) - beta*sigma**2.0                                      )
        G['mu']    = (          + beta                        +    (-1.0/sigma)*d_Lnormcdf(n_Mag) )
        G['sigma'] = (                     - beta**2.0*sigma  +  (-n_Mag/sigma)*d_Lnormcdf(n_Mag) )
    
    return [L,G]

##random number generation
def rnd_GRDR(n,para):
    
    x = np.arange(0.0,10.001,0.1)
    m = len(x)
    p_mass = np.zeros(m)
    
    x_list = np.vstack([x-0.05,x,x+0.05]).T
    p_mass = pdf_GRDR(para,x_list).dot( 0.1*np.array([1.0,4.0,1.0])/6.0 )
    
    p_mass = p_mass/p_mass.sum()
    p_mass_cum = np.hstack([0,p_mass.cumsum()])
    
    y = np.random.rand(n)
    c = np.histogram(y,p_mass_cum)[0]
    rnd = np.repeat(x,c)
    np.random.shuffle(rnd)
    
    return rnd

## calculatet the detection rate with earthquakes with M > mag_min
def calc_dtr_tv(beta,mu,sigma,mag_min):
    dtr = np.zeros_like(mu)
    for i in range(len(dtr)):
        para_bms = pd.Series({'beta':beta[i], 'mu':mu[i], 'sigma':sigma[i]})
        dtr[i] = calc_dtr(para_bms,mag_min)
    
    return dtr

def calc_dtr(para,mag_min):
    
    beta = para['beta'];  mu = para['mu'];  sigma = para['sigma'];
    
    if mag_min < mu-6.0*sigma:
        dtr = np.exp( beta*(mag_min-mu) + (beta*sigma)**2.0/2.0 )
        
    elif mag_min < mu+6.0*sigma:
        h = sigma/16.0
        x0 = mag_min
        x = np.arange(x0,mu+6.0*sigma,h)
        
        if len(x) < 3:
            x = np.array([mag_min,mag_min+h,mag_min+2.0*h])
        
        if np.mod(len(x),2) == 0:
            x = np.hstack([x,x[-1]+h])
        
        y = pdf_GRDR(para,x)
        
        w = np.ones_like(y)
        w[2:-1:2] = 2.0
        w[1::2] = 4.0
        
        Int = np.sum(y*w)*2.0*h/6.0
        Int = Int + np.exp(-beta*(x[-1]-mu)-(beta*sigma)**2.0/2.0)
        dtr = Int/np.exp(-beta*(mag_min-mu)-(beta*sigma)**2.0/2.0)
        
    else:
        dtr = 1.0;

    return dtr

##derivative of log(norm.cdf(x))
def d_Lnormcdf(x):
    return norm.pdf(x)/norm.cdf(x)

def d2_Lnormcdf(x):
    a=d_Lnormcdf(x)
    return -a*(a+x)

##########################################################
if __name__ == '__main__':
    pass