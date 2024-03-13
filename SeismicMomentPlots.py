import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
import analysisKit as ak
import matplotlib.pyplot as plt
import plotKit as pk
print("loading dataSets")
hydroSet = pd.read_csv("reformattedData/RetrofittedHydraulicDat_Apr2022.csv",
                    parse_dates = ["DateTime"],
                    index_col=["DateTime"])

events  = pd.read_csv("reformattedData/RetrofittedSeismicEvents.csv",
                    parse_dates = ["DateTime"],
                    index_col=["Stage","DateTime"]) 

def plot_M0expBarFit():
    events["M0"] = events["MomMag"].transform(ak.magToM0)
    check = ak.check_GRlaw(events, fitShift = 0, frac_trunc = 0, bin_factor = 5,scaleDown = 100000)
    fig,ax = plt.subplots(3,1)
    for i in range(3): check[0].xs(i+1).plot.bar(sharex = False,ax = ax[i],title = "stage {}".format(i+1),logy = True)
    plt.show()

def plot_multiXfitPlot():
    l = events.M0.groupby(level=0).agg(lambda c: [(x,ak.exp_lambdaFit(c,x)) for x in np.linspace(0,c.mean()/3,1000)])
    N = len(l)
    fig, ax = plt.subplots(N,1)
    [ax[i].plot(*zip(*l[i+1])) for i in range(N)]
def plot_singleXfitPlot():
    M0 = events["M0"].groupby(level=0).agg()
    M0.groupby(level=0).agg(lambda x: [(x,ak.exp_lambdaFit(x,i)) for i in np.linspace(0,M0.mean(),1000)])

events["M0"] = events["MomMag"].transform(ak.magToM0)
pk.GR_lawPlot2(events)