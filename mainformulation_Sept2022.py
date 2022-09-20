"""
This python code implements the MIP, as well as graph based approach. use this in same folder as functionsPax
"""

# First run functionsPax.py to load the functions
# run functionsPax

#%%
import pandas as pd
import csv
from datetime import datetime, date,time, timedelta
from timeit import default_timer as timer
import networkx as nx
import cplex
import gurobipy as gp
from gurobipy import *
from gurobipy import GRB
import os
import math
from matplotlib import pyplot as plt
from collections import Counter
import itertools
import numpy as np

#%%

#########################################################################
"DATA PREPROCESSING"

# Note: Write the column names in all the csv files manually
#Aircraft.csv
# Use this to get seat capacity of aircrafts given model number
data_aircraft = pd.read_csv('aircraft.csv',names=['Aircraft','Model','Family',
                                                  'Config','Dist','CostPerHour',
                                                  'TurnRound','Transit','Orig',
                                                  'Maint'], delim_whitespace = True).to_dict()
# Store only Aircraft model and seat configurations
data_aircraft = {k: data_aircraft[k] for k in set(['Aircraft','Config']) & set(data_aircraft.keys())}
# Note that ground transport has infinite capacity in our model though in the file it is -1/-1/-1
data_aircraft = ConvertAircraftToDict(data_aircraft)

#config.csv
# cost parameters and regulation costs are given here
data_config = pd.read_csv('config.csv').to_dict() #scenario parameters
# But we hardcode as a separate function
dictAllCosts = DelDownCanCosts(data_config)
# returns a dict that looks like {'Down': dictDownCosts, 'Delay': dictDelCosts,
# 'Cancel': {'A': dictCanCostsOutbound, 'R': dictCanCostsInbound}}

# the outputs are all nested dictionaries. First level keys are itin type 
# (Domestic/Cont/Intercont) and the nested keys denote the cabin class

# rotations.csv
# Ops solution gives aircraft rotations
# Given Flight 555, this tells us the aircraft flying it and date of flight
# Note: Assumed that Flight 555 flies only flies on one date. If some flight is being 
# used on comsecutive days in Ops Solution, we have to assign different flight IDs
# and only use this format of rotations.csv
data_rotations = pd.read_csv('rotations.csv', names = ['FlightNum','DepDate','Aircraft'],
                             delim_whitespace=True).to_dict()

#dist.csv
# Given any two airports, this assigns type of any flight between them,
# i.e. Domestic/Continental/Intercontinental
data_LegTypes = FindFlightTypes('dist.csv')
# If the last line of dist.csv is a # (hashtag) then there might be a stray
# node ('#',nan) in the output. We can simply ignore this
# sanity check for above dictionary: size should be #airports*(#airports-1)

# Disruption window and recovery time
# This is present in config.csv
disrupDate = date(2006,1,7)
disrupTime = time(12,0)
disrupStartTime = datetime.combine(disrupDate,disrupTime)
# The above stores date and time for disruption beginning, as a datetime object

# Also present in config.csv
recovEndDate = date(2006,1,8)
recovEndTime = time(4,0)
recovByTime = datetime.combine(recovEndDate,recovEndTime)
# stores the end of the recovery window
# hard deadline for pax to get to their destinations

# Flights.csv
flightsDataFr = pd.read_csv('flights.csv',
                            names=['Num','Orig','Dest','DepTime','ArrTime','PrevFlightBySameAircraft'],
                            delim_whitespace=True)
#Cleaning: CHeck if flights.csv has last row as #, special character,
# then drop the last row, else this causes problems later.
flightscsv = flightsDataFr[:-1].to_dict()

data_airports=[]
# List of airports stored here
data_airports.extend(flightscsv['Dest'][keys] for keys in flightscsv['Dest'].keys() if flightscsv['Dest'][keys] not in data_airports)
data_airports.extend(flightscsv['Orig'][keys] for keys in flightscsv['Orig'].keys() if flightscsv['Orig'][keys] not in data_airports)

# Call the function to make predecessor and successor for each flight
# turnaround time in mins to be passed as an argument
data_flights = PrevNextFlightList(flightscsv,data_rotations,data_LegTypes,turnTime=30)
# Now we have list of flights along with predecessor and successors, along with
# their flyng dates (based on Ops solution) and aircraft used
# (from which seating capacity can be deduced)


# recovered flights. Keep some keys as original flight data, but some flights could be missing (cancelled) 
# some flights could be new (added) or departure/arrival times are changed. Remaining capacities could also be changed
data_recovflights = data_flights # Change this manually
"""
needs to be done
"""

# Itineraries.csv
# For Itin file, headers are Ident Type Price Count (Flight DepDate Cabin)+
# Each row can have different number of columns, hence we do the conversion
# into a dictionary in a separate function
data_itin = ConvertItinToDict('itineraries.csv',data_flights) #Itinerary stored as Dict

# Note: 700/1900 itineraries had same source and destination! So we need to
# re think these, since we pruned them out. The intermediate airports in these itineraries
# have to be definitely reached in a pax solution

# Therefore data_itin will be the original pax itin. Our MIP will calculate
# recovered pax itin.

# Given an itinerary, this is data_flights with an extra key, which tells us how many
# remaining seats there are in each cabin class. Note that -1 means infinite capacity (ground transport)
data_flights_withRemCapacity = CabinCapacity(data_flights,data_aircraft,data_itin)

# Feeding initial pax itin gives us zero remaining capacity, we need to check after disruption

MaxLegNumIncrease = 2 # Bound how many legs the recovered pax itinerary has more
# than the original pax itinerary. Pax don't like it if you add more than 2 legs
#We don't end up using it, but can be given as cutoff argument when we create the networkx
# path list inside the function CreatingGraphGivenAnItinerary


# Store a DAG for every itinerary. Dict is used, each key is an itinerary,
# each key value is a DAG for the itinerary, created using source/sink knowledge
# and the flights recovered ops solution
#%%
"""
Implement preprocessing- Create the graphs Gk
"""

allAirportDAGs = {} # e.g. {1: [{'CDG':['JFK','BOS'], 'JFK':['LHR','SFO']}] , 2: ...}
# for key 1 (itinerary 1), keyvalue is a dict itself. In this nested dict, 
# 'CDG':['JFK','BOS'] means the arcs CDG -> JFK and CDG -> BOS are in the DAG

allFlightDAGs = {} # e.g. {1: [ {('CDG','JFK'):['545','67'],('JFK','LHR'):['1','2'] }], 2:...}
# For key 1 (itinerary 1), keyvalue is a dict itself. In this nested dict,
# nested key is a tuple ('CDG','JFK') is an arc of the DAG served by flights 545 and 67

allPathsDAGs = {} # key j is itinerary j. keyvalue is a nested dicti. for e.g.
# keyvalue for itin j looks like {0:['BIQ','CDG','ORY'],1:['BIQ','CDG','NTE'],2...}
# list of all patha from BIQ to ORY
timInit = timer()


totalItin = len(data_itin['Num']) # Number of itineraries
# Build the DAGs for every itinerary
for k in range(0,totalItin):
    if k%50 == 0: print('loop is in iteration',k)
    allAirportDAGs[k],allFlightDAGs[k],allPathsDAGs[k] = CreatingGraphGivenAnItinerary(data_itin,k,data_flights,data_airports,disrupStartTime,recovByTime,MaxLegNumIncrease)
# allPathsDAGs tell us the size of the network for each k
    
# Apr 25 2022 335 sec (5.5 mins)
# Jul 28 2022 427 sec (7 mins)
tim2 =  timer() - timInit


#%%
"""
MIP implementation
"""
timeLim = 1e4
# Create a new Gurobi MIP model
mod = gp.Model("mip1")

# set time limit
mod.setParam(GRB.Param.TimeLimit, timeLim)

# Number of flow conservation constraints
numflowConstraints = 0
for k in range(0,totalItin):
    # number of nodes in the DAG, subtract 1 to remove the source airport
    numflowConstraints += len(allAirportDAGs[k].keys()) - 1
    
        
# First how many variables do we have?
cabinTypes = ['F','B','E'] # types of cabin classes

totalFlightNum = len(data_flights['Num'])

# how many y_{fmk}, number of (integer) variables in MIP
# Apr 25th 2022- 3,02,6016 (whew!)
sizVarMIP = totalItin * len(cabinTypes) * totalFlightNum


numVar = 0 # number of variables in our problem

# this dict stores y_{fmk}. The keys are [f,m,k], values are Gurobi variable type Integer
yVar = {}
# dicts for storing corresponding downgrading, delay and cancellation costs, leys are [f,m,k]
cDown = {}
cDelay = {}
cCancel = {}


for m in cabinTypes: # for each cabin class (pax recovery solution)
    for k in range(0,totalItin): # for each itinerary

        ArcSetOfItin = allFlightDAGs[k] # The DAG of our itinerary
        # Number of (unique) vertex pairs with arcs between them in the DAG, not counting multiple arcs between two nodes
        NumArcsInItin = len(ArcSetOfItin.keys()) 
        
        # sink airport of this itinerary k
        itinSink = data_itin['SinkAirport'][k]
        
        
        cCancel[k] = ObjCancCost(k,data_recovflights,data_itin,data_LegTypes,dictAllCosts)
        
        
        for arc in ArcSetOfItin.keys(): # for each arc i -> j
            flightsInThisArc = ArcSetOfItin[arc]
            
            # # For each flight between the given arc, use flight ID '343' to get row number in flightdata
            # flightRowNumbers = [FindFirstKeyGivenValue(data_recovflights['Num'],v) for v in flightsInThisArc]
            
            # wee only want to use flights with non zero remaining capacity to form the network
            flightsWithNonZeroCap = []
            # for each flight
            for f in flightsInThisArc:
                
                # row number in recovered flight data for flight f
                row = FindFirstKeyGivenValue(data_recovflights['Num'],f)
                # capacity of the flight f
                upb = data_recovflights['RemCabinCapacityGivenItin'][row][m]
                
                if (upb == -1) or (upb>0):
                    if upb == -1: # ground transport have -1 as capacity, we set upper bound to infinity here
                        # this flight has some capacity
                        flightsWithNonZeroCap += [f]
                    
                        # set variable names, e.g. y-Flt4498-ArcBIQ->CDG-CabinE-Itin0
                        yVar[f,m,k] = mod.addVar(vtype=GRB.INTEGER,lb = 0, ub = GRB.INFINITY ,name="y"+"Flt%s-Arc%s->%s-Cabin-%s-Itin-%d" % (f,arc[0],arc[1],m,k))
            
                    # only bother defining y variables if there is remaining capacity left at all
                    if upb > 0:
                        # this flight has non-zero capacity
                        flightsWithNonZeroCap += [f]
            
                        # set variable names, e.g. y-Flt4498-ArcBIQ->CDG-CabinE-Itin0
                        yVar[f,m,k] = mod.addVar(vtype=GRB.INTEGER,lb = 0, ub = GRB.INFINITY ,name="y"+"Flt%s-Arc%s->%s-Cabin-%s-Itin-%d" % (f,arc[0],arc[1],m,k))
                    
                    # downgrading costs 
                    cDown[f,m,k] = ObjDowngradingCost(f,m,k,data_recovflights,data_itin,data_LegTypes,dictAllCosts)
                    
                    # delay costs
                    # only if flight has destination as sink of itinerary
                    if (arc[1] == itinSink): 
                        
                        # compute delay costs
                        cDelay[f,m,k] = ObjDelayCost(f,m,k,data_recovflights,data_itin,data_LegTypes,dictAllCosts)
                        
                    else:
                        # if flight does not have sink as destination
                        cDelay[f,m,k] = 0
                    
                
# Apr 25 2022: we prune to only use flights with non zero remaining capacity: numVar = 168006 for
# 1659 itineraries, 608 flights, 35 airports, 3 cabin classes



#%%
"Algorithm"



            