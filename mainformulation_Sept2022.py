"""
This python code implements the MIP, as well as graph based approach. use this in same folder as functionsPax
"""

#%%
import pandas as pd
import csv
from datetime import datetime, date,time, timedelta
from timeit import default_timer as timer
import networkx as nx
import gurobipy as gp
from gurobipy import *
from gurobipy import GRB
import os
import math
from matplotlib import pyplot as plt
from collections import Counter
import itertools
import numpy as np
from copy import deepcopy

#%%
"""
Set the current directory for data files from 32 possible datasets in the ROADEF challenge
"""

# name of the files used for the simulations, 
# get current working directory
parent_dir = str(os.getcwd())
# get csv files in subdirectory for this setup   

# all 32 data files
all_dataFiles = sorted(glob.glob(parent_dir+'/DATA_ROADEF2009/*'))

# 10 A instances
files_Ainstances = sorted(glob.glob(parent_dir+'/DATA_ROADEF2009/A_instances/*'))
#10 B instances
files_Binstances = sorted(glob.glob(parent_dir+'/DATA_ROADEF2009/B_instances/*'))
# 12 X instances
files_Xinstances = sorted(glob.glob(parent_dir+'/DATA_ROADEF2009/X_instances/*'))

# first A dataset
currentDir = files_Ainstances[0]

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




#%%
#########################################################################
"DATA PREPROCESSING"

# First run functionsPax.py to load the functions
# run functionsPax
from functionsPax_Sept2022 import *

# generate random recovery file
from generateRandomRecovery import *

defaultDate = '07/01/06'

GenerateRandomFlightRecoveryFile(currentDir,defaultDate)


# Note: Write the column names in all the csv files manually
#Aircraft.csv
# Use this to get seat capacity of aircrafts given model number
data_aircraft = pd.read_csv(currentDir+'/aircraft.csv',names=['Aircraft','Model','Family',
                                                  'Config','Dist','CostPerHour',
                                                  'TurnRound','Transit','Orig',
                                                  'Maint'], delim_whitespace = True).to_dict()
# Store only Aircraft model and seat configurations
data_aircraft = {k: data_aircraft[k] for k in set(['Aircraft','Config']) & set(data_aircraft.keys())}
# Note that ground transport has infinite capacity in our model though in the file it is -1/-1/-1
data_aircraft = ConvertAircraftToDict(data_aircraft)

#config.csv
# cost parameters and regulation costs are given here
data_config = pd.read_csv(currentDir+'/config.csv').to_dict() #scenario parameters
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
data_rotations = pd.read_csv(currentDir+'/rotations.csv', names = ['FlightNum','DepDate','Aircraft'],
                             delim_whitespace=True).to_dict()

#dist.csv
# Given any two airports, this assigns type of any flight between them,
# i.e. Domestic/Continental/Intercontinental
data_LegTypes = FindFlightTypes(currentDir+'/dist.csv')
# If the last line of dist.csv is a # (hashtag) then there might be a stray
# node ('#',nan) in the output. We can simply ignore this
# sanity check for above dictionary: size should be #airports*(#airports-1)

# stores the end of the recovery window
# hard deadline for pax to get to their destinations

# Flights.csv
flightsDataFr = pd.read_csv(currentDir+'/flights.csv',
                            names=['Num','Orig','Dest','DepTime','ArrTime','PrevFlightBySameAircraft'],
                            delim_whitespace=True)
#Cleaning: Check if flights.csv has last row as #, special character,
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
# their flying dates (based on Ops solution) and aircraft used
# (from which seating capacity can be deduced)

"""
Recovered Flights
"""
# recovered flights. Keep some keys as original flight data, but some flights could be missing (cancelled) 
# some flights could be new (added) or departure/arrival times are changed. Remaining capacities could also be changed
# data_recovflights = data_flights # Change this manually


## we only kept some rows of OG flight set
recovflightsDataFr = pd.read_csv(currentDir+'/random_recovered_flights.csv',
                            names=['Num','Orig','Dest','DepTime','ArrTime','PrevFlightBySameAircraft'],
                            delim_whitespace=True)
#Cleaning: CHeck if flights.csv has last row as #, special character,
# then drop the last row, else this causes problems later.
recovflightscsv = recovflightsDataFr[:-1].to_dict()

# data_airports=[]
# # List of airports stored here
# data_airports.extend(flightscsv['Dest'][keys] for keys in flightscsv['Dest'].keys() if flightscsv['Dest'][keys] not in data_airports)
# data_airports.extend(flightscsv['Orig'][keys] for keys in flightscsv['Orig'].keys() if flightscsv['Orig'][keys] not in data_airports)

# Call the function to make predecessor and successor for each flight
# turnaround time in mins to be passed as an argument
data_recovflights = PrevNextFlightList(recovflightscsv,data_rotations,data_LegTypes,turnTime=30)
# Now we have list of flights along with predecessor and successors, along with
# their flyng dates (based on Ops solution) and aircraft used
# (from which seating capacity can be deduced)


# Itineraries.csv
# For Itin file, headers are Ident Type Price Count (Flight DepDate Cabin)+
# Each row can have different number of columns, hence we do the conversion
# into a dictionary in a separate function
data_itin = ConvertItinToDict(currentDir+'/itineraries.csv',data_flights) #Itinerary stored as Dict

# Therefore data_itin will be the original pax itin. Our MIP will calculate
# recovered pax itin.

# Given an itinerary, this is data_recovflights with an extra key, which tells us how many
# remaining seats there are in each cabin class. Note that -1 means infinite capacity (ground transport)
# also outputs set of disrupted itineraries
data_recovflights,Kdis = CabinCapacity(data_recovflights,data_flights,data_aircraft,data_itin)

# dict of per pax cancellation cost, key is itin index in Kdis, key values are costs
dictKdisCancelCosts = {}
for k in Kdis:

    dictKdisCancelCosts[k] = ReturnCancelCostOfanItin(k,data_itin,dictAllCosts)

# sort Kdis by decreasing cancellation costs
Kdis.sort(key = lambda x: dictKdisCancelCosts[x],reverse=True)

remCap = deepcopy(data_recovflights['RemCabinCapacityGivenItin'])


# Feeding initial pax itin gives us zero remaining capacity, we need to check after disruption

MaxLegNumIncrease = 2 # Bound how many legs the recovered pax itinerary has more
# than the original pax itinerary. Pax don't like it if you add more than 2 legs
#We don't end up using it, but can be given as cutoff argument when we create the networkx
# path list inside the function CreatingGraphGivenAnItinerary

# list of all flights in this data set
listAllFlights = list(data_recovflights['Num'].values())

dictFlightDests = {}
dictFlightArrTime = {}
dictFlightDepDate = {}

for flight in listAllFlights:
    
    findex = FindFirstKeyGivenValue(data_recovflights['Num'],flight)
    dictFlightDests[flight] = data_flights['Dest'][findex]
    dictFlightArrTime[flight] = data_flights['ArrTime'][findex]
    dictFlightDepDate[flight] = data_flights['FlyingDate'][findex]
    



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

totalKDis = len(Kdis) # number of disrupted itineraries
# Build the DAGs for every itinerary in disrupted set
count = 0
for k in Kdis:
    if count%50 == 0: 
        print('computing Gk for itinerary %d of %d disrupted itineraries' % (count,totalKDis))
    count += 1
    
    allAirportDAGs[k],allFlightDAGs[k],allPathsDAGs[k] = CreatingGraphGivenAnItinerary(data_itin,k,data_recovflights,data_airports,disrupStartTime,recovByTime,MaxLegNumIncrease)


    if not any(allAirportDAGs[k]): 
        # dummy flight source to sink, used for cancellation
        allAirportDAGs[k][data_itin['SourceAirport'][k]] = [data_itin['SinkAirport'][k]]
        allFlightDAGs[k][(data_itin['SourceAirport'][k],data_itin['SinkAirport'][k])] = ['dummy']
# allPathsDAGs tell us the size of the network for each k
    
# Sep 22 2022: 941 out of 1626 disrupted itinearies, time was 127 sec
tim2 =  timer() - timInit

#%%

# # dict storing list of itineraries which have a given flight f in its arc set in cabin class m with some remaining capacity
# dictFlightItinMap = {}

# for flt in listAllFlights:
#     for m in cabinTypes:
        
#         # row number in recovered flight data for flight f
#         row = FindFirstKeyGivenValue(data_recovflights['Num'],flt)
#         # capacity of the flight f
#         remCap = data_recovflights['RemCabinCapacityGivenItin'][row][m]
        
        


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
for k in Kdis:
    # number of nodes in the DAG, subtract 1 to remove the source airport
    numflowConstraints += len(allAirportDAGs[k].keys()) - 1
    
        
# First how many variables do we have?
cabinTypes = ['F','B','E'] # types of cabin classes

totalFlightNum = len(data_flights['Num'])

# how many y_{fmk}, number of (integer) variables in MIP
# Apr 25th 2022- 3,02,6016 (whew!)
sizVarMIP = totalKDis * len(cabinTypes) * totalFlightNum


numVar = 0 # number of variables in our problem

# this dict stores y_{fmk}. The keys are [f,m,k], values are Gurobi variable type Integer
yVar = {}
# dicts for storing corresponding downgrading, delay and cancellation costs, leys are [f,m,k]
cDown = {}
cDelay = {}
cCancel = {}

# initialize the three terms in the objective function
objDelayTerm = 0
objDownGradingTerm = 0
objCancelTerm = 0

# dict storing list of itineraries which have a given flight f in its arc set in cabin class m with some remaining capacity
dictFlightItinMap = {}

for k in Kdis: # for each itinerary 

    yVar[k] = mod.addVar(vtype=GRB.INTEGER,lb = 0, ub = GRB.INFINITY ,name="y"+"DummyFlt-Itin-%d" % (k))
    
    ArcSetOfItin = allFlightDAGs[k] # The DAG of our itinerary
    # Number of (unique) vertex pairs with arcs between them in the DAG, not counting multiple arcs between two nodes
    NumArcsInItin = len(ArcSetOfItin.keys()) 
    
    # sink airport of this itinerary k
    itinSink = data_itin['SinkAirport'][k]
    
        
    # only bother defining y variables if there is remaining capacity left at all
    
    # define cancellation cost for itinerary k
    cCancel[k] = ObjCancCost(k,data_recovflights,data_itin,data_LegTypes,dictAllCosts)
    
    objCancelTerm += LinExpr(cCancel[k]*yVar[k])
    
    # dictionary of number of pax coming into node j, keys are each node, key values are number of pax sum_{i,m} y_{ijmk} for in node to j
    inFlowPaxNumDict = dict.fromkeys(list(allAirportDAGs[k].keys()),0)
    
    # dictionary of number of pax going out of node i, keys are each node, key values are number of pax sum_{j,m} y_{ijmk} for every out node from i
    outFlowPaxNumDict = dict.fromkeys(list(allAirportDAGs[k].keys()),0)
    
    

    for arc in ArcSetOfItin.keys(): # for each arc i -> j
        flightsInThisArc = ArcSetOfItin[arc]
        
        # node i of arc
        firstNode = arc[0]
        # node j of arc
        secondNode = arc[1]
        
        # # For each flight between the given arc, use flight ID '343' to get row number in flightdata
        # flightRowNumbers = [FindFirstKeyGivenValue(data_recovflights['Num'],v) for v in flightsInThisArc]
        
        # if there is only dummy flight, no feasible flights in this itin
        if flightsInThisArc == ['dummy']:
            
            
            # make it empty so it doesn't enter next loop
            flightsInThisArc = []
            flightsWithNonZeroCap = []
        
        # for each flight
        for f in flightsInThisArc:
              # we only want to use flights with non zero remaining capacity to form the network
            flightsWithNonZeroCap = []
            
            for m in cabinTypes: # for each cabin class (pax recovery solution)
                                
                # row number in recovered flight data for flight f
                # remCap = deepcopy(data_recovflights['RemCabinCapacityGivenItin'])
                
               # row = FindFirstKeyGivenValue(data_recovflights['Num'],f)
                # capacity of the flight f
                upb = data_recovflights['RemCabinCapacityGivenItin'][f][m]
                
                if (upb == -1) or (upb>0):
                    if upb == -1: # ground transport have -1 as capacity, we set upper bound to infinity here
                        # this flight has some capacity
                        flightsWithNonZeroCap += [f]
                        
                        # itin k contains flight f in its arcset with non zero capacity in cabin class m
                        append_value(dictFlightItinMap,(f,m),k)
                    
                        # set variable names, e.g. y-Flt4498-ArcBIQ->CDG-CabinE-Itin0
                        yVar[f,m,k] = mod.addVar(vtype=GRB.INTEGER,lb = 0, ub = GRB.INFINITY ,name="y"+"Flt%s-Arc%s->%s-Cabin-%s-Itin-%d" % (f,firstNode,secondNode,m,k))
            
                    # only bother defining y variables if there is remaining capacity left at all
                    if upb > 0:
                        # this flight has non-zero capacity
                        flightsWithNonZeroCap += [f]
                        
                        # itin k contains flight f in its arcset with non zero capacity in cabin class m
                        append_value(dictFlightItinMap,(f,m),k)
            
                        # set variable names, e.g. y-Flt4498-ArcBIQ->CDG-CabinE-Itin0
                        yVar[f,m,k] = mod.addVar(vtype=GRB.INTEGER,lb = 0, ub = upb ,name="y"+"Flt%s-Arc%s->%s-Cabin-%s-Itin-%d" % (f,firstNode,secondNode,m,k))
                    
                    # cDown[f,m,k] = ObjDowngradingCost(f,m,k,data_recovflights,data_itin,data_LegTypes,dictAllCosts)
                    
                    # downgrading costs 
                    if (ObjDowngradingCost(f,m,k,data_recovflights,data_itin,data_LegTypes,dictAllCosts) > 0):
                        
                        cDown[f,m,k] = ObjDowngradingCost(f,m,k,data_recovflights,data_itin,data_LegTypes,dictAllCosts)
                        
                        objDownGradingTerm += LinExpr(cDown[f,m,k],yVar[f,m,k])
                    
                    # delay costs
                    # only if flight has destination as sink of itinerary
                    if (secondNode == itinSink): 
                        
                        # compute delay costs
                        cDelay[f,m,k] = ObjDelayCost(f,m,k,data_recovflights,data_itin,data_LegTypes,dictAllCosts)
                        
                        objDelayTerm += LinExpr(cDelay[f,m,k],yVar[f,m,k])
                        
                    # Constraints
                    
                    # # add flow contraint term if secondNode is not sink
                    else: 
                        
                        inFlowPaxNumDict[secondNode] += yVar[f,m,k]

                    outFlowPaxNumDict[firstNode] += yVar[f,m,k]
                  
    # source airport of itinerary k
    itinSource = data_itin['SourceAirport'][k]  
    # intermediate nodes of the DAG for itin k      
    intermediateNodes = list(outFlowPaxNumDict.keys())
    
    if any(intermediateNodes):
        intermediateNodes.remove(itinSource)
    
    for anyNode in intermediateNodes:
        
        # flow constraint for intermediate node
        mod.addConstr(inFlowPaxNumDict[anyNode] == outFlowPaxNumDict[anyNode],"c1")
        
    # second constraint for source node
    mod.addConstr(outFlowPaxNumDict[itinSource] + yVar[k] == data_itin['PaxCount'][k],"c2")
 
# write fourth constraint
# for each (f,m) pair with remaining capacity
for (f,m),itList in dictFlightItinMap.items():
    
    # row number in recovered flight data for flight f
    row = FindFirstKeyGivenValue(data_recovflights['Num'],f)
    # capacity of the flight f
    remCap = data_recovflights['RemCabinCapacityGivenItin'][f][m] 
    
    # # if remaining capacity is -1, it is infinity
    # if remCap == -1:
    #     mod.addConstr(quicksum(yVar[f,m,k] for k in itList) <= GRB.INFINITY, "c4")
        
    if remCap >0 :
        
        # sometimes itList is a int, chamge to list then
        itList = [itList] if isinstance(itList, int) else itList
        
        mod.addConstr(quicksum(yVar[f,m,k] for k in itList) <= remCap, "c4")
    
# update the model  
mod.update()  
                        
# define the objective term of the MIP
totalObjTerm = objDelayTerm + objDownGradingTerm + objCancelTerm

                
# Apr 25 2022: we prune to only use flights with non zero remaining capacity: numVar = 168006 for
# 1659 itineraries, 608 flights, 35 airports, 3 cabin classes

# size(cCancel) = 1659
# size(CDelay) = 32263
# size(cDown) = 7382
# not bad, easier to compute objective  

# set objective
mod.setObjective(totalObjTerm,GRB.MINIMIZE)

mod.update()
    
# initialize output variables
out = 0 # output, the runtime
time_terminate = 0 # indicator for exceeding time threshold
optVal = 0 # optimal value

#%%
# Optimize model
mod.optimize()

# run time
out = mod.RunTime

# if we have an optimal solution
if mod.status == GRB.OPTIMAL:
    # record the optimal solution
    optVal = mod.objVal

# if the given time limit is exceeded
if mod.status==GRB.TIME_LIMIT:
    # set indicator variable to 1
    time_terminate = 1

optVal,out,time_terminate

yOpt = mod.x
yZero = yOpt.count(0.0)

ySize = mod.NumVars
# mod.computeIIS() 
# mod.write("MIP1.ilp")

#%%

with open(currentDir+"/AllDetailsAboutMIPSolution.txt", "a") as somf:

    print('Total itineraries %d' % totalItin, file = somf)
    print('Disrupted itineraries %d' % len(Kdis), file = somf)
    print('Total #flights %d' % totalFlightNum, file = somf)
    print('Preprocessing time %f' % tim2, file = somf)
    print('MIP runtime %f' % out, file = somf)
    print('MIP objective %f' % optVal, file = somf)
    print('MIP number of zero variables %f out of %d' % (yZero,ySize), file = somf)

#%%

"""
Implement Multi-Label Shortest Paths
"""

import queue

def pruneDominators(labels,numPaxItin):
    
    maxLabelperNode = min(10,numPaxItin) # max labels you consider per node, prune the rest
    n = len(labels)
    keep = [True] * n
    for i in range(n):
        for j in range(i):
            fc1, sc1, _,_ = labels[i]
            fc2, sc2, _,_ = labels[j]
            # Strict pruning
            if (fc1 <= fc2 and sc1 < sc2) or (fc1 < fc2 and sc1 <= sc2):
                keep[j] = False
    new_labels = [labels[i] for i in range(n) if keep[i]]
    
    new_labels2 = [] #output
    
    # don't keep ridiculous numbers of almost equivalent labels
    if len(new_labels) > maxLabelperNode :
        new_labels.sort(key = lambda x: (x[1], x[0]))
        
        NumSoFar = 0
        
        for lab in new_labels:
            
            _,_,_,fouc = lab
            
            NumSoFar += fouc
            
            if NumSoFar < maxLabelperNode:
                
                new_labels2.append(lab) 
                
            else:
                break
    
        return new_labels2
                
            
            
    # Sanity assertion
    #if random.random() < 0.1:
    #    print(labels, new_labels)
  #  assert len(new_labels) >= 1
    return new_labels

def mlsp(k_dis):
    mlsp_start = timer()
    cabin_classes = ['F', 'B', 'E']
    Krecov = {k:[] for k in k_dis}
    it = 1
    
    costitink = dict.fromkeys(k_dis,0.0)
    
    remCap = deepcopy(data_recovflights['RemCabinCapacityGivenItin'])
    
    for k in k_dis:
        # set of nodes of the graph
        
        graph = allAirportDAGs[k]
        visited = {}
        labels = {}

        for key in graph.keys():
            visited[key] = False
            labels[key] = []

        src = data_itin['SourceAirport'][k]
        dst = data_itin['SinkAirport'][k]
        ref_cabin_class = data_itin['ItinRefCabinClass'][k]
        ref_flight_type = data_itin['ItinRefFlightType'][k]
        original_legs = data_itin['NumOfLegs'][k]
        numPax = data_itin['PaxCount'][k]
        
        origArrTime = ConvertFlightTime(data_itin['ItinEndTime'][k], data_itin['ItinEndDate'][k])
        
        delCostforitin = dictAllCosts['Delay'][ref_flight_type]

        # this should have been in the graph
        graph[dst] = []
        visited[dst] = False
        labels[dst] = []

        q = queue.Queue()
        visited[src] = True
        # [ (#legs, downCost,flights/cabin class aka path so far) ]
        """
        add flight cap
        """
        # first comp fc = num of legs
        # second comp sc = downcosts (delay to be added later)
        # third comp tc = flights and legs used
        # fourth comp fouc = num of pax that can be carried for this label
        labels[src] = [(0, 0, [],data_itin['PaxCount'][k])]
        q.put(src)
        bfs_timer_start = timer()
        #print("Starting BFS for itin", k)
        while not q.empty():
            ap = q.get()
            #print(ap, end = " ")
            for nextap in graph[ap]:

                # continue BFS
                if not visited[nextap]:
                    q.put(nextap)
                    visited[nextap] = True
                
                for flight in allFlightDAGs[k][(ap, nextap)]:
                    
                    if flight == 'dummy': continue
                
                    flight_type = data_LegTypes[(ap, nextap)]
                    for cc in cabin_classes:
                        for fc, sc, tc, fouc in labels[ap]:
                            sc2 = sc
                            try:
                                exFlNum = data_itin['LegFlightNum'][k].index(flight)
                                old_cc = data_itin['LegCabinClass'][k][exFlNum]
                            except ValueError:
                                exFlNum = None
                            if exFlNum:
                                # check for downgrade
                                if (old_cc, cc) in dictAllCosts['Down'][flight_type]:
                                    # why are these single-element lists lol
                                    sc2 = sc + dictAllCosts['Down'][flight_type][(old_cc, cc)][0]
                            else:
                                if (ref_cabin_class,cc) in dictAllCosts['Down'][ref_flight_type]:
                                    sc2 = sc + dictAllCosts['Down'][ref_flight_type][(ref_cabin_class,cc)][0]
                            fc2 = fc + 1
                            # Don't keep if number of legs is above max
                            if fc2 <= original_legs + MaxLegNumIncrease:
                                
                                newtc = tc.copy()
                                newtc.append((flight,cc))
                                
                                newCap = remCap[flight][cc]
                                
                                if dictFlightDests[flight] == dst:
                                    
                                    newarrTime = ConvertFlightTime(dictFlightArrTime[flight],dictFlightDepDate[flight])
                                    
                                    delMins = (newarrTime-origArrTime).total_seconds()
                                    
                                    
                                    if delMins > 0:
                                        
                                        
                                        delCost = delCostforitin[cc][0]*(delMins/60.0)
                                        sc2 += delCost
                                
                                # how many pax can this label carry
                                # ignore -1 case
                                if newCap >= 0:
                                    fouc = min(fouc,newCap)
                                    
                                # only create label if it can carry >0 pax
                                if fouc>0:
                                
                                    labels[nextap].append((fc2, sc2, newtc,fouc))
                                    
                labels[nextap] = pruneDominators(labels[nextap],numPax)
                
                
        """
        allocate pax for this itin
        """
        # print(labels[dst])        
        
        # # Why are there no labels with nonzero downgrading?
        # for label in labels[dst]:
        #     if label[1] > 0:
        #         print(label,k)
        
        # labels of dst       
                
        sinkLabels = labels[dst]
    
        sinkLabels.sort(key = lambda x: x[1])
        
        if not sinkLabels: 
            costitink[k] =  dictKdisCancelCosts[k]*numPax
            Krecov[k] = [(('dummy','E'),numPax)]
            
        else:
            
            PaxsoFar = 0 # pax in this itin taken care of so far in recov solution
            remPax = numPax
            
            # all the flights and cabins actuallt used
            flightsCabinsUsed = []

            for lab in sinkLabels:
            
                _,labcost,labpath,labpax = lab
                
                paxTobeSent= min(remPax,labpax)
                # add paths (flights + leg cabin classes)
                Krecov[k].append((labpath,paxTobeSent))
                
                remPax -= paxTobeSent
                
                remCap = UpdateRemFlightCap(labpath,paxTobeSent,remCap)
                
                # store costs of sending these pax
                costitink[k] += labcost*paxTobeSent
                
                if remPax == 0:
                    break
            
            
            if remPax > 0:
                Krecov[k].append((('dummy','E'),remPax))
                
                costitink[k] += dictKdisCancelCosts[k]*remPax
            
        
        # update rem flight capacities
        # done in the loop
        
        
        bfs_timer_end = timer()
        if it % 50 == 0:
            print("Time for %ith itinerary %i = %fs" % (it, k, bfs_timer_end - bfs_timer_start))
            print("Total time so far %fs" % (bfs_timer_end - mlsp_start))
        it += 1

    return Krecov,costitink

tim3 = timer() - timInit

"""
Sort Kdis by cancellation costs, and use that as input
"""
# run the algo
KrecovMLSP, recovCostsMLSP = mlsp(Kdis)
tim4 = timer() - timInit
print("Time taken for mlsp is", tim4-tim3)

totalMLSPcost = sum(recovCostsMLSP)
# %%


# for k in Kdis[0:10]:
    
#     print("itin %d had %d pax" % (k,data_itin['PaxCount'][k]))
#     print("recovered plan for this itin is ",KrecovMLSP[k],"with cost",recovCostsMLSP[k],"\n")