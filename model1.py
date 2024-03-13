import pandas as pd
#from plotKit import *
import analysisKit as anKit
import dataKit as dk
import matplotlib.pyplot as plt
print("loading dataSets")
hydroSet = pd.read_csv("reformattedData/RetrofittedHydraulicDat_Apr2022.csv",
                    parse_dates = ["DateTime"],
                    index_col=["DateTime"])

seisSet  = pd.read_csv("reformattedData/SeismicEvents.csv",
                       parse_dates = ["DateTime"],
                       index_col=["DateTime"]) 
seisSet.plot()
dk.inaugurateStageIndex(df_hy.copy(),["Injected_L","AccumulatedM0"])
pd.DataFrame.join
#add latency
seisSet["Latent_h"]=seisSet.reset_index(level = 2)["DateTime"].groupby("Stage").diff().dt.total_seconds()/3600
#add seismic momentFraction of Expected CumulativeMagnitude
seisSet["M0"] = seisSet["MomMag"].apply(anKit.magTo_M0)
seisSet[["CumulMag","CumulM0"]] = seisSet[["MomMag","M0"]].groupby(level=0).cumsum()
seisSet["Mom/h"] = seisSet["MomMag"]/seisSet["latency_h"]
sMets = seisSet.reset_index(level = 2)[["M0","MomMag","CumulMag","CumulM0","InjectedGal","DateTime"]].groupby(level=0).agg(
    max_M0=('M0','max'),    
    max_MomMag=('MomMag','max'),
    last_M0Sum=('M0','last'),
    last_MagSum=('CumulMag','last'),
    last_Gal=("InjectedGal",'last')) 

seisSet["Normalized Magnitude"] = seisSet["CumulMag"].groupby(level=0).apply(lambda x:x/sMets["last_MagSum"][x.name])
seisSet["Normalized M0"] = seisSet["M0"].groupby(level=0).apply(lambda x:x/sMets["last_M0Sum"][x.name])
seisSet["Normalized Volume"] = seisSet["InjectedGal"].groupby(level=0).apply(lambda x:x/sMets["last_MagSum"][x.name])
seisSet["Fractional Accum MomMag"] = seisSet["Normalized Magnitude"]/seisSet["Normalized Volume"]
seisSet["Fractional Accum M0"] = seisSet["Normalized M0"]/seisSet["Normalized Volume"]
seisSet["(Normalized MomMag)/h"] = seisSet["Normalized Magnitude"]/seisSet["latency_h"]
seisSet["(Normalized M0)/h"] = seisSet["Normalized M0"]/seisSet["latency_h"]

def plotSlipVsMoment():
    seisSet.xs(1).reset_index().plot(kind = 'scatter', x="Fractional Accum MomMag",y="(Normalized MomMag)/h", logy=True)
    plt.show()
def plotSlipVs():
    seisSet.xs(1).reset_index().plot(kind = 'scatter', x="Fractional Accum M0", y="(Normalized M0)/h", logy=True)


