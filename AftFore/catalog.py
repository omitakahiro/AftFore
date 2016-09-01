import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

import datetime,numbers,sys,pickle,io

############################################
## Class
############################################
class Data_Eq(pd.core.frame.DataFrame):
    
    @property
    def _constructor(self):
        return Data_Eq
    
    def search(self,txm):
        return search(self,txm)
        
    def extract_aft(self,ms,alpha=None,xy=None,duration=32.0,fore_mc=10.0):
        return extract_aft(self,ms,alpha=alpha,xy=xy,duration=duration,fore_mc=fore_mc)
    
    def save_Data(self,file_name): 
        save_Data(self,file_name)
    
    def find_ms(self):
        return find_ms(self)
    
    def mfd(self,txm=None,mag_bin=np.arange(-2.05,10.06,0.1)):
        return mfd(self,txm=txm,mag_bin=mag_bin)
        
    def mfd_cum(Data,txm=None,mag=np.arange(-2.05,10.0,0.1),line=False):
        return mfd_cum(Data,txm=txm,mag=mag,line=line)
        
    def plot_mfd(self,txm=None,pdf=False):
        plot_mfd(self,txm=txm,pdf=pdf)
        
    def plot_mt_aft(self,txm=None,pdf=False):
        plot_mt_aft(self,txm=txm,pdf=pdf)
    
    def plot_map(self,txm=None,ms=None,aftzone=False,pdf=False):
        plot_map(self,txm=txm,ms=ms,aftzone=aftzone,pdf=pdf)
        
    
############################################
## Data Input & Output
############################################
def load_Data(file_name,dt_origin=None):
    
    if file_name[-4:] in ['.dat','.txt']:
        Data = load_dat(file_name)
        
    else:
        if file_name[-6:] == '.hinet':
            Data = load_Hinet(file_name)
        elif file_name[-4:] == '.jma':
            Data = load_JMA(file_name) 
        elif file_name[-4:] == '.pkl':
            Data = load_pickle(file_name)
        else:
            sys.exit('invalid file name')

        if dt_origin is None:
            dt_origin = Data.index[0]
        elif isinstance(dt_origin,datetime.datetime):
            pass
        elif dt_origin is 'ms':
            dt_origin = Data['Mag'].argmax()
        else:
            sys.exit('invalid dt_origin')

        Data['T'] = (Data.index - dt_origin).total_seconds()/24.0/60.0/60.0
    
    return Data

def save_Data(Data,file_name): 
    with open(file_name,'w') as f:
        for i in range(len(Data)):
            [t,mag,x,y,z] = Data.iloc[i][['T','Mag','X','Y','Z']].values
            f.write('%.7f %4.1f %8.4f %8.4f %7.2f\n'%(t,mag,x,y,z))
            
##########################
def load_dat(file_name):
    data = pd.DataFrame(np.loadtxt(file_name)[:,:5],columns=['T','Mag','X','Y','Z'])
    return Data_Eq(data)

##########################
def load_pickle(file_name):
    data = pickle.load(open(file_name,'rb'))
    return Data_Eq(data)

##########################
def date_parser_JMA(s):
    return datetime.datetime.strptime(s,'%Y %m %d %H %M %S.%f')

def load_JMA(file_name):
    df = pd.read_table(file_name,delim_whitespace=True,header=None,index_col=0,parse_dates=[[0,1,2,3,4,5]],date_parser=date_parser_JMA)
    df = df.iloc[:,:4].sort_index()
    df.columns = ['X','Y','Z','Mag']
    df.index.name = 'DT'
    return Data_Eq(df)

##########################
def date_parser_Hinet(s):
    return datetime.datetime.strptime(s,'%Y-%m-%d %H:%M:%S.%f')

def load_Hinet(file_name):
    df = pd.read_table(file_name,delim_whitespace=True,header=None,index_col=0,parse_dates=[[0,1]],date_parser=date_parser_Hinet)
    df = df.iloc[:,:8].sort_index()
    df.columns = ['Er_T','Y','Er_Y','X','Er_X','Z','Er_Z','Mag']
    df.index.name = 'DT'
    return Data_Eq(df)

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

def extract_aft(Data,ms,alpha=None,xy=None,duration=32.0,fore_mc=10.0):
    
    t_ms = ms.name; mag_ms = ms['Mag']; x_ms = ms['X']; y_ms = ms['Y']; z_ms = ms['Z'];
    
    Data = Data.copy()
    Data['T'] = ( Data.index - t_ms ).total_seconds()/60.0/60.0/24.0
    t0 = 30.0/60.0/60.0/24.0
    
    if alpha is not None:
        xy = aftzone_US(ms,alpha)
        x = xy['x']; y = xy['y'];
    elif xy is not None:
        x = xy['x']; y = xy['y'];
        
    txm_fore = {'t':[-30.0,-t0],   'x':x, 'y':y, 'mag_min':fore_mc}
    txm_aft  = {'t':[t0,duration], 'x':x, 'y':y}
    
    Data_ms   = pd.DataFrame([[0.0,mag_ms,x_ms,y_ms,z_ms]],columns=['T','Mag','X','Y','Z'],index=[t_ms])
    Data_fore = search(Data,txm_fore)
    Data_aft  = search(Data,txm_aft)
    Data_seq = pd.concat([Data_fore,Data_ms,Data_aft])
    
    return [Data_seq,{'x':x,'y':y}]

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

############################################
## mainshock and aftershock
############################################
def find_ms(Data):
    ms = Data.loc[Data['Mag'].argmax()].copy()
    return ms

def parse_str(s):
    [dt1,dt2,x,y,z,mag] = s.split()
    dt = datetime.datetime.strptime(dt1 + ' ' + dt2,"%Y-%m-%d %H:%M:%S.%f")
    ms = pd.Series([float(mag),float(x),float(y),float(z)],index=['Mag','X','Y','Z'])
    ms.name = dt
    return ms

def aftzone_US(ms,alpha):
    x_ms = ms['X']; y_ms = ms['Y']; mag_ms = ms['Mag'];
    r = 0.01*10**(0.5*mag_ms-1.8)*alpha/2.0
    x = [x_ms-r,x_ms+r]; y = [y_ms-r,y_ms+r];
    return {'x':x, 'y':y}

############################################
## Graph
############################################
def plot_mfd(Data,txm=None,pdf=False):
        
    [mag,c]         = mfd(Data,txm=txm)
    [mag_cum,c_cum] = mfd_cum(Data,txm=txm,line=True)

    plt.figure(figsize=(8.27,11.69), dpi=100)
    plt.plot(mag,c,'ko')
    plt.plot(mag_cum,c_cum,'k-')

    plt.yscale('log')
    plt.xlabel('magnitude')
    plt.ylabel('number')
    plt.xlim([-2,9])
    plt.ylim([0.5,c_cum.max()*2.0])
    plt.xticks(np.arange(-2,10))

    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes', titlesize=12)
    mpl.rc('pdf',fonttype=42)
    plt.tight_layout(rect=[0.1,0.3,0.9,0.7])

    if pdf is not False:
        if pdf is True:
            plt.savefig('mfd.pdf')
        else:
            plt.savefig(pdf)

def plot_mt_aft(Data,txm=None,pdf=False):
    
    Data_s = search(Data,txm=txm)
    
    plt.figure(figsize=(8.27,11.69), dpi=100)
    mpl.rc('font', size=12, family='Arial')
    mpl.rc('axes',linewidth=1,titlesize=12)
    mpl.rc('pdf',fonttype=42)
    
    plt.plot(Data_s['T'],Data_s['Mag'],'o',mfc=[1,0.6,0.6],mec=[1,0.6,0.6])
    plt.plot([0.125,0.125],[-1,8],'k:',dashes=(1,1))
    plt.plot([0.25,0.25],  [-1,8],'k:',dashes=(1,1))
    plt.plot([0.5,0.5],    [-1,8],'k:',dashes=(1,1))
    plt.plot([1.0,1.0],    [-1,8],'k:',dashes=(1,1))
    plt.plot([2.0,2.0],    [-1,8],'k:',dashes=(1,1))
    plt.plot([4.0,4.0],    [-1,8],'k:',dashes=(1,1))
    plt.plot([8.0,8.0],    [-1,8],'k:',dashes=(1,1))
    plt.plot([16.0,16.0],  [-1,8],'k:',dashes=(1,1))
    plt.xscale('log')
    plt.xlabel('time after the main shock [day]')
    plt.ylabel('magnitude')
    plt.ylim([-1,8])
    plt.xlim([5e-4,31])
    
    plt.tight_layout(rect=[0.2,0.2,0.8,0.8])

    if pdf is not False:
        if pdf is True:
            plt.savefig('mt_aft.pdf')
        else:
            plt.savefig(pdf)

def plot_map(Data,txm=None,ms=None,aftzone=False,pdf=False):
    
    from mpl_toolkits.basemap import Basemap
    
    #setting
    if ms is not None:
        x_ms = ms['X']; y_ms = ms['Y']; mag_ms = ms['Mag'];    
    
    if (txm is not None) and ( 'x' in txm and 'y' in txm ):
        Data_s = search(Data,txm=txm)
        [_,_,x,y,_,_] = expand_txm(txm)
        x1 = x[0]; x2 = x[1]; y1 = y[0]; y2 = y[1]   
    elif aftzone is True:
        t_ms = ms.name
        r_us = 0.01*10**(0.5*mag_ms-1.8)
        x1 = x_ms - 2.5*r_us; x2 = x_ms + 2.5*r_us;
        y1 = y_ms - 2.5*r_us; y2 = y_ms + 2.5*r_us;
        txm = {'dt':[t_ms,t_ms+datetime.timedelta(days=365.0)], 'x':[x1,x2], 'y':[y1,y2]}
        Data_s = search(Data,txm)
    else:
        Data_s = search(Data,txm=txm)
        x1 = Data_s['X'].min(); x2 = Data_s['X'].max();
        y1 = Data_s['Y'].min(); y2 = Data_s['Y'].max();
    
    map_cntr = {'x':(x1+x2)/2.0,'y':(y1+y2)/2.0}
    r = max([(x2-x1)/2.0,(y2-y1)/2.0])*1.05
    map_x1 = map_cntr['x']-r; map_x2 = map_cntr['x']+r;
    map_y1 = map_cntr['y']-r; map_y2 = map_cntr['y']+r; 
    
    #map
    fig = plt.figure(figsize=(8,8), dpi=100)
    mpl.rc('font', size=10, family='Arial')
    mpl.rc('axes',titlesize=8)
    mpl.rc('pdf',fonttype=42)
    
    step_list = np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0,20.0,50.0])
    step_grid = step_list[np.abs(step_list-(2.0*r)/5.0).argmin()]
    
    gs = mpl.gridspec.GridSpec(10,10)
    #fig.subplots_adjust(left=0.05,right=0.95,bottom=0.05,top=0.95)
    
    ax = plt.subplot(gs[0:7,0:7])
    m = Basemap(projection='cyl',llcrnrlat=map_y1,urcrnrlat=map_y2,llcrnrlon=map_x1,urcrnrlon=map_x2,resolution='i')
    m.drawcoastlines( linewidth=0.5, color='k' )
    m.fillcontinents(color='#eeeeee',lake_color='#ffffff')
    m.drawparallels(np.arange(0,90,step_grid),labels=[1,0,0,0])
    m.drawmeridians(np.arange(0,180,step_grid),labels=[0,0,1,0])
    m.plot(Data_s['X'],Data_s['Y'],'o',mfc=[0,0,0],mec=[0,0,0],markersize=2,latlon=True)
    m.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'b-')
    
    if ms is not None:
        m.plot(x_ms,y_ms,'*',mfc=[1,1,0],markersize=20,latlon=True)
        
        if aftzone is True:
            for alpha in [1.0,2.0,3.0,4.0]:
                ax.add_patch(mpl.patches.Rectangle([x_ms-0.5*alpha*r_us,y_ms-0.5*alpha*r_us],alpha*r_us,alpha*r_us,fill=False,edgecolor='r',lw=1,linewidth=2.0))
    
    ax = plt.subplot(gs[0:7,7:10])
    ax.xaxis.tick_top()
    plt.plot(Data_s['Z'],Data_s['Y'],'o',mfc=[0,0,0],mec=[0,0,0],markersize=2)
    plt.ylim([map_y1,map_y2])
    plt.yticks([])
    
    plt.subplot(gs[7:10,0:7])
    plt.plot(Data_s['X'],-Data_s['Z'],'o',mfc=[0,0,0],mec=[0,0,0],markersize=2)
    plt.xlim([map_x1,map_x2])
    plt.xticks([])
    
    if pdf is not False:
        if pdf is True:
            plt.savefig('map.pdf')
        else:
            plt.savefig(pdf)

############################################
## Aftershocks
############################################
def aftfore_data():
    
    ##main shock information
    ms_s = '2016-04-16 01:25:05.47 130.763 32.7545 12.45 7.3'
    ms = parse_str(ms_s)
    print(ms)
    
    Data = load_Data('kuma.jma')
    
    Data.plot_map(ms=ms,aftzone=True,pdf='hypo_check.pdf')
    Data_seq = Data.extract_aft(ms,alpha=3.0,duration=1.0,fore_mc=6.0)
    Data_seq.plot_map(ms=ms,pdf='hypo.pdf')
    Data_seq.plot_mt_aft(pdf='mt.pdf')
    Data_seq.save_Data('kumamoto.dat')
    
