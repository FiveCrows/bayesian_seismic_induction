import pandas as pd
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator as kme
from scipy.optimize import minimize_scalar 
from scipy.optimize import minimize
from lifelines import KaplanMeierFitter
from scipy.stats import pareto
from scipy.interpolate import CubicSpline
from scipy.interpolate import Akima1DInterpolator as akima
from scipy.interpolate import make_lsq_spline as bspline
from scipy.signal import spline_filter as sf
from scipy.signal import deconvolve
from scipy.optimize import minimize_scalar 
import matplotlib.pyplot as plt
sm_Granite = 24*10^9


def hydroSet():
    return pd.read_csv("reformattedData/RetrofittedHydraulicDat_Apr2022.csv",
                    parse_dates = ["DateTime"],
                    index_col=["Stage", "DateTime"])

def seisSet():
    return pd.read_csv("reformattedData/RetrofittedSeismicEvents.csv",
                    parse_dates = ["DateTime"],
                    index_col=["Stage","DateTime"]) 
    
def magToM0(x):
    return 10**(3*(x+6.033)/2)

def M0toMag(x):
    return np.log10(x)*2/3-6.003

#class omoriKernal:

def omoriKernal(t,t_a, R):
    #return t
    pass
    
def exp_lambdaFit(x,offset=0):
    shift = x[x>offset]    
    N=len(shift)   
    if N ==0: return np.NaN
    return N/(shift.sum()-offset*N)

def pareto_fit(x,offset=0,x_m = None):    
    shift = x[x>offset]     
    n=len(shift)
    return n/(np.log(shift).sum()-n*np.log(x_m))


def pareto_empericalErr(x,est_km,x_m,offset=0):    
    a=pareto_fit(x,offset,x_m=x_m)
    mean_err = abs(x.mean() - a*x_m/(a-1))
    f = pow(x_m/(x),a)
    #plt.plot(est_km)
    err = sum((((est_km.values)/f- 1))*mean_err)
    return(err)


def paretoAlpha(x,offset=0):
    x = x.sort_values().values
    #factor = minimize_scalar(lambda x2: pareto_meanErr(x, offset = offset,x_m = x2*x.mean()), bounds = [0,0.5], method = 'bounded').x
    km = KaplanMeierFitter()
    km.fit(x)    
    km = km.predict(x)    
    print(km)
    mn = np.mean(x)
    factor=0.25 
    x_m,offset = minimize(lambda z: pareto_empericalErr(x,km, x_m = z[0],offset = z[1]), x0=(factor,0),method = 'Nelder-Mead',options = {'disp':True} ).x
    #factor,offset = minimize(lambda z: , x0=[0.3,100000],method = 'Nelder-Mead',options = {'disp':True} ).x
    a=pareto_fit(x,offset,x_m)
    f = pow(x_m/(x),a)
    plt.plot(x,km)     
    plt.plot(x,f)
    plt.yscale('log')
    plt.ylim(0,1)   
    plt.show()
    alpha = pareto_fit(x,offset,x_m = x_m)
    return (x_m, alpha)

def getSurvivals(mag):
    km = KaplanMeierFitter()
    km.fit(mag)    
    return mag.apply(km.predict) 

def exp_dist(x,decay, offset = 0):
    return decay*np.exp(-decay*x+offset)

#def exp_cdf(decay, interval):
    #return ((exp_dist(interval.left,decay)-exp_dist(interval.right,decay))/decay)
    
    #def kap_meier(series):
    
def hyperExpSurvival(x,l1,l2):    
    m = x.mean()
    a = (l1*l2*m-l1)/(l2-l1)
    print(a)
    return(a*(np.exp(-l1*x))+(1-a)*np.exp(-l2*x))

def fit_hyperExponential(x,offset=0):
    x = x.sort_values().values
    #factor = minimize_scalar(lambda x2: pareto_meanErr(x, offset = offset,x_m = x2*x.mean()), bounds = [0,0.5], method = 'bounded').x
    km = KaplanMeierFitter()
    km.fit(x)    
    km = km.predict(x)    
    print(km)
    mn = np.mean(x)
    factor=0.25 
    l1 = exp_lambdaFit(x)
    #l2 = exp_lambdaFit(x,x.mean())cn0tc r/m/-hyperExp)/km), x0=(l1,1000),method = 'Nelder-Mead',options = {'disp':True})


    
def cubicSpline(timeSeries):
    ts = timeSeries
    ts.cut_index = (m0l_cut.index - m0l_cut.index[0]).seconds
    l_cut = m0l_cut.iloc[m0l_cut.index.drop_duplicates(keep = 'first')]
    
def crudeS_eff(df): 
    #return linregress(df.Injected_L.iloc[7000:7500],df.AccumulatedM0.iloc[7000:7500]).slope
    return 777477



def GSeff(x):
    return x.AccumulatedM0.max()/(x.Injected_L.max()-x.Injected_L.min())

def bAndMc(M0):    
    optFit = minimize_scalar(lambda x2: -exp_lambdaFit(M0,x2))
    return (-1/optFit.fun,optFit.x)
    
def bWithMc(MomMag,Mc):
    return 1/(np.mean(MomMag[MomMag>Mc])-Mc)
    
def seisIndex(L,MomMag,Mc):
    b= bWithMc(MomMag,Mc)
    N = len(MomMag[MomMag>Mc])
    dV = (L.max()-L.min())
    return np.log10(N)-np.log10(dV) + b*Mc

def maxList(S):
    l = min(S)
    mList = []
    for e in S:
        l = max(l,e)
        mList.append(l)
    return mList

def ST_omoriMetric(events):
    cross = events[['MomMag','Y','X','Depth']].reset_index()#important columns
    cross = cross.merge(cross, on  = 'Stage', how = 'outer')#outer product
    cross = cross[cross["DateTime_x"]>cross["DateTime_y"]]# drop duplicate XY YX pairs
    cross["d_time"] = cross["DateTime_x"]-cross["DateTime_y"]
    cross["d_space"] = np.sqrt((cross.Y_y-cross.Y_x)**2+(cross.X_x-cross.X_y)**2 +(cross.Depth_x-cross.Depth_y)**2)
    cross["d_mag"] = cross.MomMag_x-cross.MomMag_y
    
#def plot_multi_mMax(hydroSet,events):
if True:
    Mc = -0.5
    events = seisSet()
    ##### add injection volume to the earthquakes####
    M0 = events.M0
    M0_s = events.M0.groupby(level=0)
    events = pd.merge_asof(events.reset_index().sort_values("DateTime"),hydroSet().reset_index(1),on="DateTime").set_index(["Stage","DateTime"]).sort_index()
    ev_s = events.groupby(level=0)
    ################################################################
    #do max calcs
    ##################################
    
    
    events["max"] = M0_s.transform(lambda x: M0toMag(maxList(x)))
    
    ######
    events["max_pred1"] = ev_s.apply(lambda x: M0toMag(2*sm_Granite*abs(x.Injected_L-x.Injected_L[0])).droplevel(0))
    events["max_pred2"] = ev_s.apply(lambda x: M0toMag(GSeff(x)*abs(x.Injected_L-x.Injected_L[0])).droplevel(0))
    events["max_pred3"] = ev_s.apply(lambda x: ((np.log10(abs(x.Injected_L-x.Injected_L[0]))+seisIndex(x.Injected_L,x.MomMag,Mc))/bWithMc(x.MomMag,Mc)).droplevel(0))
    events["max_pred4"] = ev_s.apply(lambda x: M0toMag(-x.AccumulatedM0+2*GSeff(x)*abs(x.Injected_L-x.Injected_L[0])).droplevel(0))
    
    noStage = events.copy(deep = True).reset_index().set_index("DateTime")
    L0 = events.Injected_L[0]
    si = seisIndex(noStage.Injected_L,noStage.MomMag,Mc)
    b = bWithMc(noStage.MomMag,Mc)
    print("b = {}".format(b))
    noStage["max"] = maxList(noStage.MomMag)
    noStage["max_pred1"] = noStage.apply(lambda x: M0toMag(2*sm_Granite*abs(x.Injected_L-L0)),axis = 1)
    noStage["max_pred2"] = noStage.apply(lambda x: M0toMag(GSeff(events)*abs(x.Injected_L-L0)),axis = 1)
    noStage["max_pred3"] = noStage.apply(lambda x: (np.log10(abs(x.Injected_L-L0))+Mc)/b,axis = 1)
    noStage["max_pred4"] = noStage.apply(lambda x: M0toMag(-x.AccumulatedM0+2*GSeff(events)*abs(x.Injected_L-L0)),axis = 1)
    noStage["max_predBest"] = noStage.apply(lambda x: np.log())
    
    ###########
    ###start plotting
    ###############
    #########    
    day0 = events.index[0][1]
    fig, ax = plt.subplots(1,3,sharey = True)
    fig2,ax2 = plt.subplots()
    labels = ["max","max_pred1","max_pred2","max_pred3","max_pred4"]
    colors = ["black", "blue", "purple", "green", "orange"]
    pltDat = events.reset_index().groupby("Stage")
    noStage["Flow"] = abs(noStage.Injected_L.diff()).cumsum()
    
    ## NonStagedPlot
    
    vSort = noStage.sort_values("Injected_L")    
    [noStage[l[0]].plot(ax = ax2) for l in zip(labels,colors)]
    plt.show()
    
    [ax2.scatter(vSort.Injected_L,vSort.MomMag,color="orange", alpha = 0.2)]

    for p in zip(ax,pltDat, colors):
        dat = p[1][1]
        h=(dat.DateTime-day0).dt.total_seconds()/(360*60)
        p[0].scatter(h,dat["MomMag"],color = "orange", alpha = 0.2)  
        maxI = dat["MomMag"].idxmax()
        p[0].scatter(h[maxI],dat["MomMag"][maxI])            
        for l in zip(labels,colors):
            #against time
            p[0].plot(h,dat[l[0]],label = l[0], color = l[1])  
    ax[0].set_ylabel("Max Moment Magnitude")
    ax2.set_ylabel("Max Moment Magnitude")
    ax[0].set_xlim(1.5,3.9)
    ax[1].set_xlabel("Hour")
    ax2.set_xlabel("Liters")
    ax2.legend()
    ax[0].legend()    
    fig.tight_layout()
    plt.show()
    np.log((1-(-stage1.Injected_L + stage1.Injected_L.max())/stage1.Injected_L.max() )*len(stage1[stage1.MomMag>-0.88])/np.log(1/0.5))/3.516 -0.88
    events.MomMag.merge()
    mag.merge(mag, on = MomMag)
    
    pd.DataFrame.merge
    pd.Timedelta.seconds