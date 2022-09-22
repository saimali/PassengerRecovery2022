#%%
import pandas as pd
import numpy as np
import random
from functionsPax_Sept2022 import *
from datetime import date, timedelta
import fileinput

#%%
# disruption date
defaultDate = '07/01/06'

#%%
def randomDelay(dummy):
    
    # positive delay minutes
    posDelaynumber = timedelta(hours=np.random.randint(0,7), minutes=np.random.randint(0,60))
    # no delay
    noDelay = timedelta(hours=0, minutes=0)
    
    # choose a delay time among two options
    twoDelayOptions = [posDelaynumber,noDelay]
    
    # return no delay wp 0.4, some pos delay wp 0.6
    return random.choices(twoDelayOptions,[0.4,0.6])[0]

def addTime(inputTime, delay):

    if pd.isna(inputTime):
        return inputTime

    inputTimeConverted = ConvertFlightTime(inputTime, defaultDate)
    inputTimeDelayed = inputTimeConverted + delay
    dayDiff = (inputTimeDelayed.date() - inputTimeConverted.date()).days
    suffix = ""
    if dayDiff > 0:
        suffix += "+" + str(dayDiff)
    newTime = inputTimeDelayed.strftime("%H:%M") + suffix 
    return newTime

def GenerateRandomFlightRecoveryFile(currentDir,defaultDate):

    flightsDataFr = pd.read_csv(currentDir+'/flights.csv',
                                names=['Num','Orig','Dest','DepTime','ArrTime','PrevFlightBySameAircraft'],
                                delim_whitespace=True)
    
    bad_rows = flightsDataFr['Num'].str.contains('#')
    
    flightsDataFr = flightsDataFr[~bad_rows]
    
    
    flightsDataFr['Delay'] = flightsDataFr['Orig'].apply(randomDelay)
    
    flightsRecovered = flightsDataFr.sample(frac=0.8)
    flightsRecovered['DepTime'] = flightsRecovered.apply(lambda x: addTime(x['DepTime'], x['Delay']), axis=1)
    flightsRecovered['ArrTime'] = flightsRecovered.apply(lambda x: addTime(x['ArrTime'], x['Delay']), axis=1)
    
    flightsRecovered.drop('Delay', axis=1, inplace=True)
    flightsRecovered.to_csv(currentDir+'/random_recovered_flights.csv', index=False, header=False, sep=" ")
    
    with open(currentDir+"/random_recovered_flights.csv","a") as myfile:
        myfile.write("#")
        
    return 