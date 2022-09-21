import pandas as pd
import numpy as np
from functionsPax_Sept2022 import *
from datetime import date, timedelta

defaultDate = '07/01/06'

def randomDelay(dummy):
    return timedelta(hours=np.random.randint(0,4), minutes=np.random.randint(0,60))

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

flightsDataFr = pd.read_csv('flights.csv',
                            names=['Num','Orig','Dest','DepTime','ArrTime','PrevFlightBySameAircraft'],
                            delim_whitespace=True)

flightsDataFr['Delay'] = flightsDataFr['Orig'].apply(randomDelay)

flightsRecovered = flightsDataFr.sample(frac=0.8)
flightsRecovered['DepTime'] = flightsRecovered.apply(lambda x: addTime(x['DepTime'], x['Delay']), axis=1)
flightsRecovered['ArrTime'] = flightsRecovered.apply(lambda x: addTime(x['ArrTime'], x['Delay']), axis=1)

flightsRecovered.drop('Delay', axis=1, inplace=True)
flightsRecovered.to_csv('random_recovered_flights.csv', index=False, header=False, sep=" ")