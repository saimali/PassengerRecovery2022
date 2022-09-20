" All classes and functions defined here "
import pandas as pd
import math
import csv
from datetime import datetime,date,time,timedelta # for datetime objects
import timeit
import networkx as nx
# import cplex

" Dictionary functions "
# Find the first matching key given a value in a dictionary
def FindFirstKeyGivenValue(input_dict,value):
    return next((k for k,v in input_dict.items() if v== value), None)

# Find all matching keys as a list given a value in a dictionary
def FindAllKeysGivenValue(input_dict,value):
    return list({k for k, v in input_dict.items() if v== value})

# function returning the intersection of two lists
def ListIntersection(lst1,lst2):
    
    # use of hybrid method
    tempset = set(lst2)
    lst3 = [value for value in lst1 if value in tempset]
    return lst3

" Date and Time Functions "
# Function that converts a flight time like 00:30 into my datetime format
# e.g. 11:30 = 11h,30m,00 sec and 00:30+1 = 00h 30m 00 sec on next day
# Date of the lfight can be given in format 19/12/06 DD/MM/YY
def ConvertFlightTime(xTime,xDate): 
    outTime = datetime.strptime(xDate + xTime[:5], '%d/%m/%y%H:%M') # concatenate date and time and store it in datetime format
    # if time is written as 00:30+1 then the date should be moved to next day
    if "+" in xTime:
        extraDay = int(xTime[-1]) # Usually +1 day, i.e. same time next day
        outTime = outTime + timedelta(days = extraDay) # added to date attribute
        
    return(outTime)
        
# Function that compares two dates in DD/MM/YY format
def CompareDate_IsFirstBigger(xDate,yDate):
    x2 =   datetime.strptime(xDate, '%d%m%y')  # datetime object    
    y2 =   datetime.strptime(xDate, '%d%m%y')  # datetime object    
    
    return(x2>y2) # Boolean if x>ys

    
"""
This function makes a zero matrix.
Parameters:
n_rows: number of rows
n_columns: number of columns  
output: matrix of zeros
"""
def make_zeros(n_rows: int, n_columns: int):
    # define empty matrix
    matrix = []
    for i in range(n_rows):
        matrix.append([0.0] * n_columns)
    return matrix


" Functions used in preprocessing the data"

# Comparing two cabin classes with the ordering F>B>E
# this functions output if first arg > second arg boolean, i.e. whether first 
#cabin class > second cabin class
def CompareCabinClass(first,second):
    dictMapping = {'F':2,'B':1,'E':0} # ordering is now mapped to numbers
    return (dictMapping[first] > dictMapping[second]) # Is first > second

# From a list of cabin classes, return the highest class with the logic F>B>E
def HighestCabinClass(cabinList):
    dictMapping = {'F':2,'B':1,'E':0} # ordering is now mapped to numbers
    highestNum = max( [dictMapping[j]] for j in cabinList ) # returns [0]/[1]/[2]
    highestClass = FindFirstKeyGivenValue(dictMapping,highestNum[0]) # invert to F/B/E
    
    return(highestClass) # outputs F/B/E, don't care about the index
    
# Comparing two flight typs with the logic Domestic (D) < Continental (C) < Intercontinental (I)
# outputs boolean answer whether first > second
def CompareFlightType(first,second):
    dictMapping =  {'I':2,'C':1,'D':0} # ordering is now mapped to numbers
    return (dictMapping(first) > dictMapping(second)) # Is first > second
        
# From a list of flight type, return the highest flight type I>C>D
def HighestFlightType(FlightTypeList):
    dictMapping = {'I':2,'C':1,'D':0} # ordering is now mapped to numbers
    highestNum = max( [dictMapping[j]] for j in FlightTypeList ) # returns [0]/[1]/[2]
    highestFltType = FindFirstKeyGivenValue(dictMapping,highestNum[0]) # invert to I/C/D
    
    return(highestFltType) # outputs I/C/D, don't care about the index


" Cost parameters are defined in these functions"
# take from config.csv, easier to hard code
def DelDownCanCosts(configData):
    
    # dict of delay costs per minute- nested dict
    dictDelCosts = { 'D' : {'F':[1.25], 'B':[0.8], 'E':[0.05]},
                            'C': {'F':[1.25], 'B':[0.85], 'E':[0.15]},
                            'I':{'F':[1.25], 'B':[0.9], 'E':[0.25]} }
                            
    # cancellation costs as a nested dict for outbound costs                            
    dictCanCostsOutbound = { 'D' : {'F':[2500], 'B':[1500], 'E':[250]},
                            'C': {'F':[2750], 'B':[1750], 'E':[600]},
                            'I':{'F':[3000], 'B':[2000], 'E':[1000]} }
    
    # cancellation costs as a nested dict for inbound costs                            
    dictCanCostsInbound = { 'D' : {'F':[7500], 'B':[4500], 'E':[750]},
                            'C': {'F':[8250], 'B':[5250], 'E':[1500]},
                            'I':{'F':[9000], 'B':[6000], 'E':[3000]} }

    # downgrading costs as a nested dict                          
    dictDownCosts = { 'D' : {('F','B'):[150], ('F','E'):[200], ('B','E'):[150]},
                            'C': {('F','B'):[400], ('F','E'):[500], ('B','E'):[400]},
                            'I':{('F','B'):[750], ('F','E'):[1500], ('B','E'):[750]} }

    dictAllCosts = { 'Down':dictDownCosts, 'Delay':dictDelCosts, 
                    'Cancel' :  {'A': dictCanCostsOutbound, 'R': dictCanCostsInbound }}    
    
    # return a highly nested dictionary with all cost parameters
    return(dictAllCosts)       
    
" Remaining capacity of each flight when a given itin is implemented"
# if you pass data_flights, it should output full capacities
# if you pass data_recovflights, it should output remaining capacities
def CabinCapacity(someflightData,aircraftData,itinData):
    # Output will be modified flightData dictionary, with an extra key which tells us
    # about remaining capacities
    totalItin = len(itinData['Num'])
    # triple nested dict
    someflightData['RemCabinCapacityGivenItin'] = dict.fromkeys(someflightData['Num'].keys(),{})
    for i in someflightData['Num'].keys():
        # e.g. A380#1
        tail = someflightData['AircraftUsed'][i]
        cabCapacity = aircraftData[tail]
        # remaining capacity is initialized to aircraft capacity
        someflightData['RemCabinCapacityGivenItin'][i] = cabCapacity
        
    # For every itinerary
    for j in range(0,totalItin):
        # number of passengers
        itin_pax = itinData['PaxCount'][j]
        itin_numLegs = itinData['NumOfLegs'][j]

    for j2 in range(0,itin_numLegs):
        # e.g. Flight '343' flies this leg
        legflightID = itinData['LegFlightNum'][j][j2]
        # e.g. 'E' class
        legflightClass = itinData['LegCabinClass'][j][j2]
        
        # the row number/index of Flight 343 in flightData
        indexFlightData = FindFirstKeyGivenValue(someflightData['Num'],legflightID)
        # for ground transportation, the capacity is infinity, but it is given as -1 in the dataset
        if someflightData['RemCabinCapacityGivenItin'][indexFlightData][legflightClass] != -1:
            # number of passengers assigned to this flight by the itinerary
            allPaxAssigned = someflightData['RemCabinCapacityGivenItin'][indexFlightData][legflightClass] - itin_pax
            # If it goes below zero, then it means itinData has allocated more passenfers to this flight than the
            # remaining capacity, we simply output 0, though we should really also output that itinData should be
            # changed, it is allocating more than it should
            someflightData['RemCabinCapacityGivenItin'][indexFlightData][legflightClass] = max(0,allPaxAssigned)
    
    return(someflightData) # return a modified flightData (either original or recovered can be used in this
    # function). Now the remaining capacities are updated using the itinerary.
        
" given dist.csv, store which type among D/C/I a leg is, given source and sink "
def FindFlightTypes(fileName):
    # make it into a dict with keys given below
    simpleDict = pd.read_csv(fileName, names = ['Source','Destination','Distance','Type'], 
                             delim_whitespace=True).to_dict()
    # check if s
    out = {} # output
    for j in range(len(simpleDict['Source'].keys())): # for every pair of airports
        sourceAirp = simpleDict['Source'][j] # source
        destAirp = simpleDict['Destination'][j] # destination
        # dict with keys as tuples ('JFK','BOS') : 'D', i.e. keyvalues are flight types
        out[(sourceAirp,destAirp)] = simpleDict['Type'][j]
    return(out)

" Function that converts aircrafts.csv file into an usable dictionary "
# basicAircraftDict is a simple dict we made for aircraft with keys 'Aircraft' and 'Config', this function uses aircraft data
def ConvertAircraftToDict(basicAircraftDict):

    # Delimiter
    delimiter = '/' # / is the delimiter in config values
    out = {}
    siz = len(basicAircraftDict['Aircraft'])

    # For each aircraft
    for j in range(0,siz):
        
        tailNum = basicAircraftDict['Aircraft'][j]
        if tailNum != '#': # Sometimes file has last line as just #
            tailconfig = basicAircraftDict['Config'][j]
            tailconfigList = tailconfig.split(delimiter)
        
            out[tailNum] = {}
            out[tailNum]['F'] = int(tailconfigList[0])
            out[tailNum]['B'] = int(tailconfigList[1])
            out[tailNum]['E'] = int(tailconfigList[2])
        # e.g. out now looks like {'A318#1 : {'B':0,'E':0,'F':123}}
        
    return(out)
    
" This function creates a subdictionary of previous and next flights for every flight"
# Input is a dictionary made from a file like flights.csv
# Note: yet to add prev flight based on the last column entries in flights.csv,
# it's weird to see some turnaround times of 20 mins? e.g. row 120 in flights.csv
def PrevNextFlightList(flightData,rotationData,flightTypeData,turnTime):
    
    flightData['PrevFlights']={}
    flightData['NextFlights'] = {}
    flightData['FlyingDate'] = {}
    flightData['AircraftUsed'] = {}
    flightData['FlightType'] = {}
    
    # Use rotationData (Ops Solution) to find out the date a flight is being
    # used by Ops solution as well as the aircraft being used for it
    # Store this in flightData as well. Code below:
    for f in flightData['Orig'].keys(): # A particular flight
        # dict.keys() returns the list of keys of the dict
        
        flightID = flightData['Num'][f] # e.g. flightID could be 4336
        # Index in rotationData of Flight 4336
        indexInRotationData = FindFirstKeyGivenValue(rotationData['FlightNum'],flightID)
        # Now we find the date of this flight and aircraft used for it
        # and store it on flightData
        flightData['FlyingDate'][f] = rotationData['DepDate'][indexInRotationData]
        flightData['AircraftUsed'][f] = rotationData['Aircraft'][indexInRotationData]
    
    #Predecessors/Previous Flights and successor flights for each flight
    for f in flightData['Orig'].keys(): # A particular flight f
        
        originAirport = flightData['Orig'][f] # Origin of flight f
        destAirport = flightData['Dest'][f] # Dest of flight f
        origDepTime = flightData['DepTime'][f] # Dep Time of flight f
        origDepDate = flightData['FlyingDate'][f] # Dep date of flight f
    
        # Assign flight types for each flight D/C/I
        flightData['FlightType'][f] = flightTypeData[(originAirport,destAirport)]
        
        # First we find the predecessor flights
        for j in flightData['Dest'].keys(): # List of all flights j
            
            if flightData['Dest'][j] == originAirport: # if dest of j is orig of f
                prevArrTime = flightData['ArrTime'][j] # Arr time of j
                prevArrDate = flightData['FlyingDate'][j] # Arr time of j
                # Is ArrTime of j + turnTime < DepTime of f? Then it could be a prev
                # flight to f, so store as a predecessor
                t1 = ConvertFlightTime(prevArrTime,prevArrDate) # Datetime format
                t2 = ConvertFlightTime(origDepTime,origDepDate) # Datetime format
                if t2 > t1 + timedelta(minutes=turnTime): # j could be a prev flight of f
                    
                    flightData['PrevFlights'].setdefault(f,[]).append(j) # Add j to list of prev flights of f
                    flightData['NextFlights'].setdefault(j,[]).append(f) # Add j to list of next flights of f
                    # setdefault is to make sure key f is empty initially we don't get an error
    
    return(flightData) 

" function that converts itineraries.csv file into an usable dictionary "
# filename = itineraries.csv 
def ConvertItinToDict(fileName,flightData):
    
    # Delimiter
    filename_delimiter = ' ' #space is the delimiter in our csv files
    out = {}
    out['Num'] = {} # Itin number
    out['InOrOut'] = {} # In or outbound flight
    out['UnitCost'] = {} # Unit cost per pax in Euros
    out['PaxCount'] = {} # Number of pax in this itin
    out['NumOfLegs'] = {} # Number of flight legs for the itin
    out['LegFlightNum'] = {} # Each flight's leg number
    out['LegFlyDate'] = {} #Each leg's flying date
    out['LegCabinClass'] = {} # Each leg's cabin class
    out['SourceAirport'] = {} # Source Airport for whole itin
    out['ItinStartTime'] = {} # start time of the whole itin at source airport
    out['ItinStartDate'] = {} # start date of the whole itin at source airport
    out['SinkAirport'] = {} # Sink airport for whole itin
    out['ItinRefCabinClass'] = {} # reference cabin class of pax in this itin
    out['ItinRefFlightType'] = {} # reference flight type for pax in this itin
    out['ItinEndTime'] = {} # Whole itin end time at sink airport
    out['ItinEndDate'] = {} # Whole itin end date at sink airport
    # Delays are calculated wrt this quantity
    
    # The above are the keys to the output dict of this function
    
    # open the file and extract
    with open(fileName,'r') as temp_f:
        # Read the lines
        lines = temp_f.readlines()
        
        itinCount = 0
        for j in range(0,len(lines)-1):
            # Count the column count for the current line
            l = lines[j]
            l_split = l.split(filename_delimiter)
            siz = len(l_split)
            
            # The source flight's row index in the flightData file
            firstFlightinItin = FindFirstKeyGivenValue(flightData['Num'],l_split[4])
            # The sink flight's row index in the flightData file
            lastFlightinItin = FindFirstKeyGivenValue(flightData['Num'],l_split[1+(siz-5)])
            # Prune to only have itineraries where source and sink are not the same
            if flightData['Orig'][firstFlightinItin] != flightData['Dest'][lastFlightinItin]:
                
                # Source airport and itinerary start time
                out['SourceAirport'][itinCount] = flightData['Orig'][firstFlightinItin]
                out['ItinStartTime'][itinCount] = flightData['DepTime'][firstFlightinItin]
                out['ItinStartDate'][itinCount] = flightData['FlyingDate'][firstFlightinItin]
                
                # Sink airport and itinerary end time
                out['SinkAirport'][itinCount] = flightData['Dest'][lastFlightinItin]
                out['ItinEndTime'][itinCount] = flightData['ArrTime'][lastFlightinItin]
                out['ItinEndDate'][itinCount] = flightData['FlyingDate'][lastFlightinItin]
                
                out['Num'][itinCount] = l_split[0]
                out['InOrOut'][itinCount] = l_split[1]
                out['UnitCost'][itinCount] = float(l_split[2])
                out['PaxCount'][itinCount] = int(l_split[3])
                out['NumOfLegs'][itinCount] = int((siz-5)/3)
                
                # store cabin class of all legs
                allLegCabinClass = []
                # store flight type (D/C/I) for all legs
                allLegFlightType = []
                
                # this stores all the leg details of the itinerary
                for j2 in range(1, 1+ int((siz-5)/3)):
                    temp = 3*j2
                    out['LegFlightNum'].setdefault(itinCount,[]).append(l_split[temp+1])
                    out['LegFlyDate'].setdefault(itinCount,[]).append(l_split[temp+2])
                    out['LegCabinClass'].setdefault(itinCount,[]).append(l_split[temp+3])
                    # update list of cabin classes used in this itin
                    allLegCabinClass += l_split[temp+3]
                    # The leg flight's row index in the flightData file
                    currLegFlightIndex = FindFirstKeyGivenValue(flightData['Num'],l_split[temp+1])
                    # update list of all flight types in this itin
                    allLegFlightType += flightData['FlightType'][currLegFlightIndex ]
                
                # compute reference cabin class of this itin
                out['ItinRefCabinClass'][itinCount] = HighestCabinClass(allLegCabinClass)
                out['ItinRefFlightType'][itinCount] = HighestFlightType(allLegFlightType)
                
                itinCount += 1
    
    # Close file
    temp_f.close()
    
    return(out)
    
# From givenAirport and givenDateTime, here is a dict with keys as the possible
# airports that can be reached and the key for each possible destination is a 
# list of corresponding flights from givenAirport to that possible airport
def NextReachableAirport(givenAirport,givenDateTime,flightData,recovByTime):
    outAirports = {} # output is a dictionary of reachable airports as keys
    # along with all feasible corresponding flights as values for the key
    # e.g. { 'CDG': ['323','454'], 'LHR':['4354']}

    earliestTime = {} # output is a dicitonary of reachable airports as keys 
    # along with all earliest possible times to reach it as values for the key
    # e.g. { 'CDG': {'11:00}, 'LHR':{'13:00}}
    
    # A temporary dictionary
    tempDict = {}
    
    # All flights starting from givenAirport are stored in candidateFlights,
    # note that this stores index of flight in flights.csv not FlightNum
    candidateFlights = FindAllKeysGivenValue(flightData['Orig'],givenAirport)
    
    # Prune the list: Only those f starting after givenDateTime, else discard f
    candidateFlights = [f for f in candidateFlights if 
                        (ConvertFlightTime(flightData['DepTime'][f],flightData['FlyingDate'][f]) >= givenDateTime)]
    # this is now the list of possible flights from givenAirport that
    # start after givenDateTime
    
    # Prune the list: Only those f arriving before end of recovery window
    candidateFlights = [f for f in candidateFlights if 
                        (ConvertFlightTime(flightData['ArrTime'][f],flightData['FlyingDate'][f]) <= recovByTime)]
    # this is now the list of possible flights from givenAirport that
    # start after givenDateTime and end within the recovery window
    
    # Reminder again that f is row number of a flight in flightData and not
    # flight ID, so we ultimately store only flight IDs
    for f in candidateFlights: # for each candidate flight
        # destination of the flight f
        fDest = flightData['Dest'][f]
        # ID of the flight f, e.g. 545
        fID = flightData['Num'][f]
        
        # destination of f stored as reachable airport, as a key. Then the flights are
        # stored as a key value for this airport/
        outAirports.setdefault(fDest,[]).append(fID)
        
        # arrival time of f in DateTime format
        fArrDateTime = ConvertFlightTime(flightData['ArrTime'][f],flightData['FlyingDate'][f])
        # Note if Flying date is 15/02/2009 and ArrtIME IS '00:30+1' THIS OUTPUT 
        # will automatically be 16/02/2009 at 00:30
        
        # Same keys as above, but store arrival datetime for the airport
        # Need this for the next loop, where can can calculate earliest time
        tempDict.setdefault(fDest,[]).append(fArrDateTime)
        
    # for every reachable airport
    for airport in tempDict.keys():
        # Min arrival time among all flights to the reachable airport
        earliestTime[airport] = min(tempDict[airport])
        
    
    
    return(outAirports,earliestTime)
    
#Recursive function that builds the graph given an itinerary
# itinIndex should start from 1 ? Unclear.
def CreatingGraphGivenAnItinerary(itinData,itinIndex,flightData,allAirports,disrupStartTime,recovByTime,MaxLegNumIncrease):
# Lots of pruning happens inside.
    
    itin_Num = itinData['Num'][itinIndex] # check itineraries.csv in this row
    itin_InOrOut = itinData['InOrOut'][itinIndex] # In or Ourbound itinerary?
    itin_UnitCost = itinData['UnitCost'][itinIndex] # Unit cost per pax in Euros
    itin_PaxCount = itinData['PaxCount'][itinIndex] # Number of pax in this itin
    itin_NumOfLegs = itinData['NumOfLegs'][itinIndex] # Number of flight legs in this itin
    itin_LegFlightNum = itinData['LegFlightNum'][itinIndex] # Each leg's flight number
    itin_LegCabinClass = itinData['LegCabinClass'][itinIndex] # Each leg's cabin class
    itin_SourceAirport = itinData['SourceAirport'][itinIndex] # Source airport for whole itinerary
    itin_SinkAirport = itinData['SinkAirport'][itinIndex] # Sink airport for whole itinerary
    itin_EndTime = itinData['ItinEndTime'][itinIndex] # Whole itin end time at sink airport
    
    itin_StartTime = itinData['ItinStartTime'][itinIndex] # Whole itin start time at source airport
    itin_LegFlyDate = itinData['LegFlyDate'][itinIndex] # Each leg's flying date
    # Storing starting date and time of first leg of itinerary as a DateTime object
    # itin_LegFlyDate[0] because the flying date of the first leg only
    itinDateTime = ConvertFlightTime(itin_StartTime,itin_LegFlyDate[0])
    
    # Outputs
    airportDAG = {} # Output will be a dictionary, like an adjacency matrix
    # looks like { "a" : ["d"], "b" : ["c"], "c": ["b","c","d"], "f": []}
    flightDAG = {} # Output will be a dictionary of feasible flights in the same context.
    # given an origin and a destination airport pair as key, the values are the list of 
    # flights between them.
    
    # Initialize a list of nodes which are reachable from source or sink
    initialNodes = [itin_SourceAirport,itin_SinkAirport]
    
    # Nodes visited so far, put False for all airports in the whole network
    visited = dict.fromkeys(allAirports,False)
    airportDAG,flightDAG = GraphBuilder(itin_SourceAirport,itinDateTime,flightData,
                                        itin_SinkAirport,recovByTime,visited)
    
    # intermedNodes will only have nodes which are actually on a path from
    # source to sink airports
    
    # airportDAG and flightDAG still contain arcs which don't lead to the sink.
    # It has paths not leading to sink as well, we have to prune this using intermedNodes
    
    ######## Pruning airportDAG and flightDAG ########
    # This will have only feasible arcs (i,j) which are on some path from
    # source to sink, also removes arcs going into sourceAirport
    prunedNodeSet = set()
    for airp1 in airportDAG.keys():
        for airp2 in airportDAG[airp1]:
            prunedNodeSet.add((airp1,airp2))
            
    # Complement of feasible arcs, i.e. unwanted arcs
    unwantedNodepairs = set(flightDAG.keys()) - prunedNodeSet
    for unwanted_key in unwantedNodepairs: del flightDAG[unwanted_key]
    # Now flightDAG will only have keys of the form (JFK,BOS) if both JFK and BOS
    # are intermediate nodes on a path from source to sink
    
    # Dict of all patha from sourceAirport to sinkAirport. Keys are just indexing of paths
    pathsDAG = {}
    j = 0
    # Using networkx package's graph structure
    G = nx.DiGraph(list(flightDAG.keys()))
    
    # cutoff argument can be used to restrict the length of paths
    # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.algorithms.simple_paths.all_simple_paths.html
    allpaths = nx.all_simple_paths(G,source=itin_SourceAirport,target=itin_SinkAirport,
                                   cutoff = itin_NumOfLegs + MaxLegNumIncrease)

    # Only reachable nodes from source to sink
    reachableNodesOnly = set([itin_SourceAirport,itin_SinkAirport])                 
    prunedNodeSet2 = set()
    
    # This is O(n!) so time consuming
    for path in allpaths:
        pathsDAG[j] = path # stores all paths as a dict
        j += 1
        for first,second in zip(path,path[1:]):
            prunedNodeSet2.add((first,second)) # stores only nodes (i,j) in the path
            # reachableNodesOnly,add(second)
    
    # airportDAG and flightDAG still need pruning
    # Complement of feasible arcs, i.e. unwanted arcs: pruning
    unwantedNodepairs = set(flightDAG.keys()) - prunedNodeSet2
    for unwanted_key in unwantedNodepairs: del flightDAG[unwanted_key]
    # flightDAG only now contains proper (i,j) arcs in some path from source to sink
    
    # Use flightDAG to get new airportDAG. this gives accurate arc list
    airportDAG = {}
    # Each key is a tuple (JFK,BOS)
    # RUNTIME COMMENT: THIS LOOP SEEMS TO BE REALLY SLOW
    for tuplekey in flightDAG.keys():
        
        # {JFK:[BOS]} is assigned
        airportDAG.setdefault(tuplekey[0],[]).append(tuplekey[1])
        
    # this will output the right network, unless there are edge cases
    return(airportDAG,flightDAG,pathsDAG)


# Recursive function that buils a graph given source + time and sink    
def GraphBuilder(givenAirport,givenDateTime,flightData,sinkAirport,recovByTime,visited):  
    
    airportDict = {}
    flightDict = {}
    ## nodeList = [givenAirport,sinkAirport]
    # From givenAirports, produce dict of reachable airports
    # and corr flights, as well as dict of reachable airports and
    # earliest possible time that airport can be reached from givenAirport
    NextAirportsFlights,NextAirportEarliestTimes = NextReachableAirport(givenAirport,givenDateTime,flightData,recovByTime)
    
    # Add all the reachable airports to the key givenAirport of airportDict
    # Therefore we now know all arcs (givenAirport,dest) are in the graph
    # the argument inside extend() is to avoid duplication when extending lists
    airportDict[givenAirport] = NextAirportsFlights.keys()
    
    visited[givenAirport] = True
    # For every airport reachble from givenAirport and every flight used
    # to reach this destination,
    for dest in NextAirportsFlights.keys():
        for flgt in NextAirportsFlights[dest]:
            
            # Add the flight to the ordered pair key (givenAirport,dest) of flightDict
            # Therefore we add flgt to the list of flights between (givenAirport,dest)
            flightDict.setdefault((givenAirport,dest),[]).append(flgt)
            
    # For every airport reachable from givenAirport and every flight used
    # to reach this destination,
    for dest in NextAirportsFlights.keys():
        # for flgt in NextAirportsFlights[dest]:
            # Find next arc as long as we haven't reached sinkAirport already
        if dest != sinkAirport:
            if visited[dest] == False:
                # starting from dest, recursive call to build next airports, we use
                # earliest time dest can be reached, and only look at flights from dest
                # that start after this time
                destStartTime = NextAirportEarliestTimes[dest]
                # nodeList is a list of nodes which are in a path from givenAirport to sinkAirport
                # USeful for pruning nodes in the output which are not on a path to sinkAirport
                ##if dest not in nodeList:
                 ## nodeList.append(dest)
                # Recursive step
                ## delted Cout
                aOut,bOut = GraphBuilder(dest,destStartTime,flightData,sinkAirport,recovByTime,visited)
                airportDict.setdefault(dest,[]).extend(x for x in aOut[dest] if x not in airportDict[dest])
                flightDict.update(bOut)
                ## nodeList.extend( x for x in cOut if x not in nodeList)
        #else:
            ## deleted nodeList
            #return(airportDict,flightDict)
    visited[givenAirport] = False
     ## deleted nodeList
    return(airportDict,flightDict)
                

# Given original itinerary and recovered flight schedules and a flight to check,
# this function computes the downgrading costs for a variable y_fmk in the MIP
# Be very careful to feed recov flight data and not orig flight data
def ObjDowngradingCost(someFlight,someCabinClass,someItin,recovFlightData,origItinData,
                       flightTypeData,allCostsDict):
    
    # all the legs in the original itinerary
    legsinorigItin = origItinData['LegFlightNum'][someItin]
    
    # if someFlight was also a leg in the original itinerary,
    # In this case, downgrading cost is only if cabin class in recovered pax is
    # strictly lowered
    if someFlight in legsinorigItin:
        
        # index of leg (1st/2nd/3rd) is this flight in original itinerary
        legNumber = legsinorigItin.index(someFlight)
        # cabin class in original itin for this flight
        origClass = origItinData['LegCabinClass'][someItin][legNumber]
        
    else: # someFlight is only in recovered pax soln, not original itinerary
        # therefore we see if someFlight's cabin class is compared to reference cabin class of the itin (RHS)
        origClass = origItinData['ItinRefCabinClass'][someItin]
        
    downCost = 0 # initialize, also it is zero when the if loop below is not entered
    if CompareCabinClass(origClass,someCabinClass):
        
        # row index of the flight someFlight in the flight data file
        rowIndexInFlightData = FindFirstKeyGivenValue(recovFlightData['Num'],someFlight)
        #Use flight type data, the key is origin and destination of someFlight
        # it is an ordered pair key in flightTypeData, key value gives us D/C/I
        fgtType = flightTypeData[(recovFlightData['Orig'][rowIndexInFlightData],recovFlightData['Dest'][rowIndexInFlightData])]
        
        downCost = allCostsDict['Down'][fgtType][(origClass,someCabinClass)][0]
        
    return downCost
        
# Given original itinerary and recovered flight schedules and a flight to check,
# this function computes the delay costs for a variable y_fmk in the MIP
# Be very careful to feed recov flight data and not orig flight data
def ObjDelayCost(someFlight,someCabinClass,someItin,recovFlightData,origItinData,
                       flightTypeData,allCostsDict):
    
    # row index of the flight someFlight in the flight data file
    rowIndexInFlightData = FindFirstKeyGivenValue(recovFlightData['Num'],someFlight)
    
    # when does someFlight arrive
    someFlightArrivalTime = recovFlightData['ArrTime'][rowIndexInFlightData]
    someFlightArrivalDate = recovFlightData['FlyingDate'][rowIndexInFlightData]
    
    # original itinerary's arrival time and date
    origItinEndTime = origItinData['ItinEndTime'][someItin]
    origItinEndDate = origItinData['ItinEndDate'][someItin]
    
    t1 = ConvertFlightTime(someFlightArrivalTime,someFlightArrivalDate) # Datetime format
    t2 = ConvertFlightTime(origItinEndTime,origItinEndDate) #Datetime format
    
    delayCost = 0 # initialize, also it is zero when the if loop below is not entered
    
    if t1 > t2:
        
        # how many minutes of delay in itinerary k, if we pick someFlight in recovered itinerary for k
        delayMinutes = t1 - t2
        
        #Use flight type data, the key is origin and destination of someFlight
        # it is an ordered pair key in flightTypeData, key value gives us D/C/I
        fgtType = flightTypeData[(recovFlightData['Orig'][rowIndexInFlightData],recovFlightData['Dest'][rowIndexInFlightData])]
        
        # delay cost is linear in number of delay minutes
        delayCost = allCostsDict['Delay'][fgtType][someCabinClass][0] *int(delayMinutes.total_seconds() / 60)  
        
    return delayCost
    

# Given original itinerary and recovered flight schedules and a flight to check,
# this function computes the cancellation costs for a variable y_fmk  (but really for y_k in the MIP
# Be very careful to feed recov flight data and not orig flight data
def ObjCancCost(someItin,recovFlightData,origItinData,flightTypeData,allCostsDict):
    
    # in bound or outbound itinerary
    InorOutitin = origItinData['InOrOut'][someItin]
    
    #reference flight type
    RefType = origItinData['ItinRefFlightType'][someItin]
    
    # ref cabin class
    origClass = origItinData['ItinRefCabinClass'][someItin]
    
    CancCost = allCostsDict['Cancel'][InorOutitin][RefType][origClass]
    
    CancCost = CancCost[0] + origItinData['UnitCost'][someItin]
        
    return CancCost