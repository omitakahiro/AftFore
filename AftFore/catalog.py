import numpy as np
import pandas as pd

import datetime,numbers,sys

############################################
############################################
class Data_Eq(pd.DataFrame):
    
    @property
    def _constructor(self):
        return Data_Eq
    
    def search(self,txm):
        return search(self,txm)
    
    def mfd(self,txm=None,mag_bin=np.arange(-2.05,10.06,0.1)):
        return mfd(self,txm=txm,mag_bin=mag_bin)
        
    def mfd_cum(Data,txm=None,mag=np.arange(-2.05,10.0,0.1),line=False):
        return mfd_cum(Data,txm=txm,mag=mag,line=line)
    
############################################
## Data Input & Output
############################################
def load_dat(file_name):
    data = pd.DataFrame(np.loadtxt(file_name)[:,:2],columns=['T','Mag'])
    return Data_Eq(data)

############################################
## Search
############################################
def search(Data,txm):
    
    if txm is None:
        Data = Data.copy()
    else:
        
        [dt,t,x,y,z,mag_min] = expand_txm(txm)
        
        if dt is not None:
            Data = Data.loc[dt[0]:dt[1]].copy()
        elif t is not None:
            Data = Data.query('%f < T < %f' % (t[0]+1e-6,t[1]-1e-6)).copy()
                
        if x is not None:
            Data = Data.query('%f < X < %f' % (x[0],x[1])).copy()
        
        if y is not None:
            Data = Data.query('%f < Y < %f' % (y[0],y[1])).copy()
        
        if z is not None:
            Data = Data.query('%f < Z < %f' % (z[0],z[1])).copy()
            
        if mag_min is not None:
            Data = Data.query('%f < Mag' % mag_min).copy()
                                      
    return Data

def expand_txm(txm):
    
    dt = None; t = None; x = None; y = None; z = None; mag_min = None;
    
    ## dt
    if 'dt' in txm:
        dt = txm['dt']
        
        if isinstance(dt[0],numbers.Number) or isinstance(dt[1],numbers.Number):
            sys.exit('error dt is datetime.datetime')
        
        if isinstance(dt[0],datetime.datetime) and isinstance(dt[1],datetime.timedelta):
            dt = [dt[0],dt[0]+dt[1]]

        if isinstance(dt[0],datetime.timedelta) and isinstance(dt[1],datetime.datetime):
            dt = [dt[1]-dt[0],dt[1]]
    
    ## t
    if 't' in txm:
        t = txm['t']

        if isinstance(t,dict):
            t = [t['st'],t['en']]

    ## x
    if 'x' in txm:
        x = txm['x']

    ## y
    if 'y' in txm:
        y = txm['y']

    ## z
    if 'z' in txm:
        z = txm['z']

    ## mag_min
    if 'mag_min' in txm:
        mag_min = txm['mag_min']
    
    return [dt,t,x,y,z,mag_min]

############################################
## Basic method
############################################
def mfd(Data,txm=None,mag_bin=np.arange(-2.05,10.06,0.1)):
    
    Data_s = search(Data,txm=txm)
    Mag = Data_s['Mag'].values
    
    m = (mag_bin[:-1]+mag_bin[1:])/2.0
    c = np.histogram(Mag,mag_bin)[0]
    
    return [m,c]

def mfd_cum(Data,txm=None,mag=np.arange(-2.05,10.0,0.1),line=False):
    
    Data_s = search(Data,txm=txm)
    Mag = Data_s['Mag'].values
    
    [c,mag_bin] = np.histogram(Mag,np.append(mag,np.inf))
    c_cum = np.flipud(np.cumsum(np.flipud(c)))
    
    if line is False:
        return [mag,c_cum]
    else:
        mag_l   = np.vstack([mag_bin[:-1],mag_bin[1:]]).T.flatten()
        c_cum_l = np.repeat(c_cum,2)
        return [mag_l,c_cum_l]
