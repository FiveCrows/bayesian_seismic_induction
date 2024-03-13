import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

################################
# This script is for cleaning up seismic and hydro data before plots are made of it.
# its a little slow which is why its done in its own script.
# Much of this has to do with staging. Seismic data is in stages,
# the stages need to be added to the hydro data, and columns are added for seconds relative to the
# start of the first stage, and also from the start of the current stage
# Load and format the hydroSet

col_hydroCore = {#Datetime is index
                    "YYYY/MM/DD": "YYYY/MM/DD", 
                    "HH:MM:SS":"HH:MM:SS",
                    'Accum Flow (Gal)': 'InjectedGal',
                    'FLOW METER (GAL)': 'InjectedGal_Rate',
                    'Bit Depth (feet)': 'Depth_Ft',
                    'WELLHEAD PSI (PSI (G))': 'WellPSI',
                    }

col_seisCore = {#Datetime is index
                    "DateTime":"DateTime",
                    ' MomMag':'MomMag',                
                    '   PGV  ':'PGV',                
                    '      Y     ':'Y',
                    '      X     ':'X',
                    '    Depth   ':'Depth',
                    ' P S/N  ':'P_S/N',
                    ' S S/N  ':'S_S/N',
                    ' Stage': 'Stage',
                    ' Location   ':'Location',
                    '   Error  ':'Error',                    
                    ' Quality':'Quality', 
                    ' Cluster':"Cluster",                    
                    ' Status':"Status",                    
                    '    Profile  ':"Profile",                    
                    }

path_hydro = "data/hydroData.csv"
path_seis1 = "data/stage1Seismicity.xlsx"
path_seis2 = "data/stage2Seismicity.xlsx"
path_seis3 = "data/stage3Seismicity.xlsx"
paths_seis = [path_seis1,path_seis2,path_seis3]




def hydroSet(colSelections = col_hydroCore, path = path_hydro):
    """_summary_

    Args:
        colSelections (dict): to define which columns to choose and their names
        Defaults to col_hydroCoreions.

    Returns:
        dataframe: hydroSet renamed columns and DateTime 
        
    """
    
    print("Loading hydro dataset")
    df = pd.read_csv(path)    
    df = hydroSet[col_hydroCore.keys()].rename(columns=col_hydroCore)
    df["DateTime"] = pd.to_datetime(hydroSet[["YYYY/MM/DD", "HH:MM:SS"]].apply(" ".join, axis=1)  ) 
    df["Inject_Pa"] = df.pop("WellPSI")*6894.76
    volCols = ["InjectedGal", 'InjectedGal_Rate']
    
    df[["Injected_L","Inject_L/s"]] = df[volCols]*3.78541
    df["Inject_m"] = df["Depth_Ft"]/3.28084
    df=hydroSet.drop(["YYYY/MM/DD", "HH:MM:SS","Depth_Ft"]+volCols, axis =1).set_index("DateTime")
    df[[]]
    return df

def seisSet(colSelections = col_seisCore, p = paths_seis):
    """_summary_
    Args:
        colSelections (dict): to define which columns to choose and their names
        Defaults to col_seisCoreions.["YYYY/MM/DD", "HH:MM:SS"]

    Returns:
        dataframe: seismic dataframe with renamed columns and DateTime 
    """
    print("Loading Seismic Dataset")
    df = pd.DataFrame()
    for path in p:
        df = df.append(pd.read_excel(path,skiprows=22))
    df["DateTime"] = pd.to_datetime(df[["Origin Date",'   Origin Time  ']].apply(" ".join, axis=1))
    df = df[col_seisCore.keys()].rename(columns=col_seisCore)
    #df["StageHour"] = (df.DateTime-df.groupby("Stage").DateTime.transform('min'))/np.timedelta64(1, 'h')
    return df

def syncStages(stagedSet, unstagedSet):
    """ 
    this set mutates the unstagedSet to have matching stages,
    both sets gain time relative to tstage beginning
    Args:
        stagedSet (_type_): The stagedSet should time connected stages
        unstagedSet (_type_):This stage should have dateTime)
    """ 
    print("Synchronizing Stages")
    startTimes = stagedSet.groupby("Stage")["DateTime"].min()
    endTimes = stagedSet.groupby("Stage")["DateTime"].max()
    bins = pd.IntervalIndex.from_tuples([(x[0],x[1]) for x in zip(startTimes,endTimes)])
    unstagedSet["Stage"] = (pd.cut(unstagedSet["DateTime"],bins,include_lowest=True)).cat.codes+1
    unstagedSet["StageHour"] = (unstagedSet.DateTime-unstagedSet.groupby("Stage").DateTime.transform('min'))/np.timedelta64(1, 'h')
    #unstagedSet = unstagedSet.set_index(["Stage","StageHour"])
    return unstagedSet

def inaugurateStageIndex(df,relativeProperties):
    columns = relativeProperties+["DateTime"]        
    if index.name == "DateTime": 
        df.reset_index(inplace=True)
    df[relativeProperties+["Time_Elapsed"]] = (df[columns] - df.groupby("Stage")[columns].transform('first'))
    df.set_index("Stage", "Time_Elapsed",inplace = True)
    return df
    
    
    return df
def eventsWithDetail(events, details):    
    """
    Assuming events are stochastically dispersed in time, 
    and details are evenly dispersed
    merges using nearest dateTime, does not interpolate
    interpolation accuracy improvement is expected to be marginal for original datasets
    this may need to be changed
    assumes stages and hours

    Args:
        events: dataset containing details of seismic events
        details: dataset adding details to time periods not seismic, such as injection volume
    """    
    
    return pd.merge_asof(events.sort_index(),details,on="DateTime",left_index=True,right_index=True,)
    
    #should be paddded firs t but  whatev er, not gonna do it until its a problem
    seis["DateTime"].__round__
def env(events, details):
    """_summary_

    Args:
        events (_type_): _description_
        details (_type_): _description_
    """
# get start and end times

#def augSeisSet():



#print("hydroSet time column configured")

#hydroSet.insert(2, 'Stage', 0)




#hydroSet.insert(3, 'stageSecond', hydroSet.apply(lambda x: (
#    x.dateTime-stageStartTimes[x.Stage-1]).total_seconds(), axis=1))



#seis = eventsWithDetail(seis,hydro)
if __name__ == "__main__":
    print("Formatting hydraulic Set")
    hydro = hydroSet()
    print("Formatting seismic set")
    seis = seisSet()
    pd.merge_asof(events.reset_index(0).sort_index(),details.droplevel(0),left_index=True,right_index=True)
    seis = eventsWithDetail(seis,hydro)
    
    hydro = hydro.join(seis.set_index("DateTime")["M0"].cumsum())
    hydro["seisEvents"]
    seis.set_index("DateTime")
    #print("syncingStages")
    #seis.reset_index()
    syncStages(seis,hydro)
    seis.to_csv("data/SeismicEvents.csv")
    hydro.to_csv("data/HydraulicDetails.csv")
    
#regularEvents.groupby(level=0)["M0"].transform(lambda x: pd.cut(x,int((x.max()-x.min())/np.sqrt(x.var()))))