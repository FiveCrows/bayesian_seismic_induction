import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import numpy as np
import analysisKit as ak
from analysisKit import *
from matplotlib import ticker as mtick
from lifelines import KaplanMeierFitter
from scipy.optimize import minimize_scalar as minimize
from sksurv.nonparametric import kaplan_meier_estimator as km
#def seisPlot(seisData):

######### a utility function not even used anymore
def plotStageLines(series,ax):
    """_summary_

    Args:
        height (_type_): _description_
    """
    startTime = pd.to_datetime(series["dateTime"][0])
    stageStarts = [x.total_seconds() for x in (pd.to_datetime(stageTimes["startTime"])-startTime)]
    stageStops = [x.total_seconds() for x in (pd.to_datetime(stageTimes["endTime"])-startTime)]
    plt.vlines(stageStarts,color = 'green',ymin = 0, ymax = height, label = 'Stage_Start')
    plt.vlines(stageStops,color = 'red', ymin = 0, ymax = height, label = 'Stage_Stop')



def barLineSeismicPlotWithVolumes(df_hy):
    fig,ax = plt.subplots()
    ax.bar(df_hy.index,df_hy.Seismic_Events.diff(), width = 0.1, label = "2h Event Count")
    ax2 = ax.twinx()
    ax2.plot(df_hy.index, df_hy.AccumulatedM0, color = 'red', label = 'Accumulated Seismic Moment')
    ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    ax.set_ylabel("count")
    ax2.set_ylabel("Seismic Moment")    
    plt.show()
#compare cumulative earthquake magnitude and injection volumes
#seisSet.where(seisSet["Stage"] ==3).set_index("stageSecond")["cumMag"].plot()
def seisVolPlot():
    fig,axes = plt.subplots(3,1)
    seisSet = seisSet.set_index("StageSecond")
    hydroSet = hydroSet.set_index("stageSecond")
    tick_count = 8
    #plt.locator_params(axis ='x', nbins = tick_count)
    plt.grid(True)
    moment_T,vol_T,max_moment = [0,0,0]
    vol_T = [0,0,0]
    max_moment=[0,0,0]
    #a nice plot  for flow vs seismic moment
    for i in range(3):
        ax_ = axes[i]
        ax_.grid(b=True, which='major', color='b', linestyle='-')
        ax_.grid(visible = True)
        volume = hydroSet[hydroSet["Stage"]==i+1]["Accum Flow (Gal)"]
        volume_T = volume.iloc[-1] 
        vol_T[i]=volume_T 
        volume.multiply(1/volume_T).plot(label = "Stage {}: Cumulative Gallons".format(i+1), ax = ax_,color = "depth")
        seis = seisSet[seisSet["Stage"] ==i+1]["M_0"]
        max_moment[i] = max(seis)
        seis = seis.cumsum()
        seis_T = seis.iloc[-1]
        moment_T[i]=seis_T
        seis.multiply(1/seis_T).plot(label = "Cumulative Seismic Moment", ax = ax_, color = "depth")    
        ax_.set_xticks([round((volume.index[-1])*tick/(tick_count),1) for tick in range(tick_count)]) #tick by hours)
        ax_.set_xticklabels([round((volume.index[-1])*tick/(360*tick_count),1) for tick in range(tick_count)]) #tick by hours)
        ax_.set_xlabel("")
        ax_.legend()
        if i == 1:
            ax_.set_ylabel("Percent of Final Value", fontsize = 10)
    plt.xlabel("Hours", fontsize = 10)
    fig.suptitle(" Seismic Moment vs Volume (Cumulative)")
    plt.show()
    plt.savefig("Seismic Moment vs Volume (Cumulative)")
    #plt.show()
    
    ######################This time in terms of  slip
    slipRatio = [moment_T[i]/vol_T[i]for i  in range(3)]
    maxRatio = [max_moment[i]/vol_T[i]for i  in range(3)]
###make barplot of discretized seismic frequency count with injection volume to color
    seisSet["Seismic Interval"] = pd.cut(seisSet["MomMag"],10)
    superSet = seisSet.merge(hydroSet,how ="inner",left_on = 's_nearest', right_on = 'dateTime')
    superSet = superSet.rename(columns = {"Accum Flow (Gal)":"Volume"})
    momentAgg = superSet[superSet["Stage_x"]==2].groupby("Seismic Interval").agg({"Seismic Interval":'count',"MomMag": 'var', "Volume":'mean'})
    momentAgg = momentAgg.rename(columns = {"Volume":"avgVolume", "Seismic Interval": "Interval Count"})
    #colored bar chart 
    px.bar(momentAgg,x = momentAgg.index.astype('str'), y = "Interval Count", color  = "avgVolume",
    color_continuous_scale=px.colors.sequential.thermal).show()




#check original correlations and filter out more junk data, look for outliers, get some sense of the data
    topCorr = np.abs(superSet.corr()).sort_values("MomMag", ascending = False,axis = 1).columns
    corrSet = superSet[topCorr[:17]].drop(["Stage_x","Stage_y","Quality",'Source',"M_0","second"],axis=1)

    fig,axes = plt.subplots(3,3)
    corrSet = corrSet.assign(Stage = superSet["Stage_x"])
    #corrSet = corrSet.set_index("MomMag")
    for i in range(9):
        ax_ = axes[i%3,i//3]
        corrSet.plot.scatter(x=corrSet.columns[i+1], y="MomMag", ax=ax_,color = corrSet["Stage"], colormap = "gnuplot")
    plt.savefig("multiplot")
    plt.show()


    corrSet = np.abs(corrSet.corr()).sort_values("MomMag",ascending=False,axis=1)
    plt.figure()
    sns.heatmap(corrSet)
    plt.savefig("RawCorrelationHeatmap")

################################################################
#Find the  slip to volume ratio

#make plot of injection volume to average seismic scaleexpected

def seisHistogram():
    print("pass")
    

    
def threeStagePlot(df, type = 'line', vertical = True):
    if vertical:
        fig, ax = plt.subplots(3,1)
    else:
        fig, ax = plt.subplots(1,3)
    for i in range(3):
        subframe.xs(i+1).plot(ax=ax)
        
def plot_bVSoffset(M0):
    bVSoffset = M0.groupby(level = 0).apply(lambda x: [np.linspace(0,x.mean()*3,1000),[ak.exp_lambdaFit(x,offset) for offset in np.linspace(0,x.mean()*3,1000)]]) 
    for block in zip(bVSoffset, plt.subplots(3,1)[1]): block[0].plot(block[1][0],block[1][1])
    
def plot_bVSoffset(mag):
    survs = getSurvivals(mag).groupby(level = 0)
    fig, ax = plt.subplots(survs)
    
    
def GR_lawPlot(df, frac_trunc = 0, bin_factor = 2,scaleDown = 1000000): 
    #trunc_M0 = df[df["M0"].groupby(level = 0).transform(lambda x:x<x.quantile(0.9999-frac_trunc))]    
    M0 = df.M0
    opt_decays = M0.groupby(level=0).apply(lambda x: -minim(lambda x: -exp_lambdaFit(x1,x2).fun))
    M0_trunc = df[df["M0"].groupby("Stage").transform(lambda x:x<x.quantile(1-frac_trunc))].M0
    #### determine decay rates    
    #plot data as histogram
    hist = M0_trunc.groupby(level=0).apply(lambda x: pd.cut(x/scaleDown,int(bin_factor*np.sqrt(len(x)).round()),precision=1))
    hist.name = "Count"
    hist = pd.DataFrame(hist.groupby(level=0).apply(lambda x: x.value_counts().sort_index()))
    #fit straight exponential
    percentErr_l2 = hist.groupby(level = 0).apply(lambda x: np.sqrt(((x.exp_fit-x.Count)**2).sum()/((x.Count**2).sum())))
    
    hist_trunc = hist[hist["Count"].groupby("Stage").transform(lambda x: x<x.quantile(1-frac_trunc))]
    #hist_trunc = hist[hist["Count"].groupby("Stage").transform(lambda x: x>x.quantile(offset))]
    percentErr_l2trunc = hist_trunc.groupby(level = 0).apply(lambda x: np.sqrt(((x.exp_fit-x.Count)**2).sum()/((x.Count**2).sum())))        
    #decay_avg err = 1/decay-df.groupby("Stage")["M0"].mean()
    return [hist,decay, {"percentErr_l2":percentErr_l2, "percentErr_l2Trunc": percentErr_l2trunc, }] 


def GR_lawPlot2(df):    
    print(df)
    M0=df.M0.groupby(level=0)    
    
    N = len(M0)
    est_km = M0.apply(lambda x:kme(x==x,x))    
    decay1 = M0.apply(lambda x:ak.exp_lambdaFit(x))
    optFit= M0.apply(lambda x:minimize(lambda x2: -(ak.exp_lambdaFit(x,x2))))
    paretoFits = [0,0,0,0]
    for i in [1,2,3]: 
        print("Now optimizing stage {}!".format(i))
        paretoFits[i] = ak.paretoAlpha(df.M0.xs(i),offset =optFit[i].x) 
        print(paretoFits)
    fig, axs = plt.subplots(1,N)
    axs[0].set_ylabel("Fraction of Events")
    axs[1].set_xlabel("Minimum Moment Magnitude")
    for i in range(N):
        ax = axs[i]
        s=i+1
        x = est_km[s][0]
        y1=est_km[s][1]
        ax.plot(x,y1,label = ("Empirical Distribution"))
        #ax.plot(x,np.exp(-x*decay1[s]),label = ("Max Likelihood Fit"))
        l,dx = optFit[s][["fun","x"]]#l is negative l
        y2 = np.exp(l*(x-dx))
        ax.plot(x,y2, label =("Exponential Fit"))
        #a,dx = paretoFit[s][["fun","x"]]
        x_m = paretoFits[s][0]
        a = paretoFits[s][1]
        
        
        y3 = (x_m/(x-dx))**(a)
        print(y3)
        ax.plot(x,y3, label =("Pareto Fit"))
        ax2 = plt.twinx(ax)
        ax2.plot(x,100*(y1-y3)/y1,color = "red", label = ("exp Err") )                
        ax2.plot(x,100*(y1-y2)/y1,color = "crimson", label = ("Pareto Err") )                
        ax.set_yscale('log')
        ax.set_xlim(x[0],x[-5])
        ax.set_ylim(est_km[s][1][-2],1.1)
        ax.legend()                    
        ax2.set_ylim(0,100)
    ax2.legend()
    ax2.spines["right"].set_color('red')
    ax2.tick_params(color = 'red')
    ax2.set_ylabel("percent error")
    plt.show()
    
def grLaw_Plots(mag):
        
        mag = mag.groupby(level=0).apply(lambda x: x.sort_values().droplevel(0))#sort each stage
        surv = mag.groupby(level=0).apply(getSurvivals).rename("Emperical")
        magGroups = mag.groupby(level=0)        
        b_M = magGroups.apply(lambda stage: stage.transform(lambda M: exp_lambdaFit(stage,M))).rename('b_M')
        frame = mag.to_frame().merge(b_M, on = ['Stage','DateTime'])  
        frame["bMin_fit"]=frame.groupby(level=0).apply(lambda x: np.exp(-(x.MomMag-x.MomMag.min())*x.b_M.min())).droplevel(0)                
        frame = frame.reset_index().merge(surv,on =  "DateTime").set_index('Stage')
        bMedian = frame[frame.b_M.isin(b_M.groupby(level=0).median())].groupby(level=0).head(1)
        bMedian = bMedian.rename(columns = {'MomMag': 'M_Median', 'b_M':'b_median'})
        frame = frame.reset_index().merge(bMedian, on = 'Stage').set_index('Stage')
        frame["bMedian_fit"]=frame.groupby(level=0).apply(lambda x: np.exp(-(x.MomMag-x.M_Median)*x.b_median)).droplevel(0)                        
        frame = frame.sort_values(["Stage","MomMag"])
         frame.groupby(level=0).apply(lambda stage: exp_dist(stage.MomMag,stage.b_M.min(),stage[stage.b_M == stage.b_M.min()].MomMag.values))
        bMin = magGroups.apply(lambda x: exp_lambdaFit(x,x.min()))
        
        bTail = magGroups.apply(lambda x: exp_lambdaFit(x,(2*x.max()+x.min())/3))
        
        fig,ax = plt.subplots(3,1)
        frame[["bMin_fit","Emperical"]].xs(1).plot()
        [sub[1][1].droplevel(0).plot(ax=sub[0]) for sub in(zip(ax,frame["Emperical"].groupby(level=0)))]
        frame.merge(mag,on = "DateTime", inplace = True)
        for s in [1,2,3]:
            stage = frame.xs(s)
            #mins = stage.min()
            #bMin = stage["MomMag"].apply(lambda x: exp_lambdaFit(x,minMom))
            

        