#! /usr/local/bin/python

__author__     = 'Zach Dischner'                      
__copyright__  = "NA"
__credits__    = ["NA"]
__license__    = "NA"
__version__    = "0.0.1"
__maintainer__ = "Zach Dischner"
__email__      = "zach.dischner@gmail.com"
__status__     = "Dev"
__doc__        ="""
File name: scheduler.py
Authors:  
Created:  Jan/13/2016
Modified: Jan/13/2016

Prototype ground station scheduling library, specifically for coding challenge
put on by Planet Labs. See prompt.md for details. Should be considered a good 
first order scheduling approximation

Accesses should be stored as a csv file or equivalent, with the following properties per access

    ```
    ,antenna,eclipse_state,end_time,id,max_el,satellite,start_time,terminator_time,lost_imagery,midpoint,duration,utility
    ```

Algorithm:

Notes:

Improvements:
    * Precompute all exclusions - AKA a dictionary with {access_id: [list of access_ids to exclude]}
        * Bet this would speed up the SchedulerGraph.generate_possibilities() substantially
        * Also would build framework for Integer Optimization below
    * Use Google's GANTT chart library for visualization https://developers.google.com/chart/interactive/docs/gallery/ganttchart
    * Formulate as an Integer Optmization Problem https://developers.google.com/optimization/mip/integer_opt
    * Better Schedule Deconflicter
Examples:

"""

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import json
import logging
import pandas as pd
import random
import plotly.plotly as py
import plotly
from plotly.tools import FigureFactory as ff
import coloredlogs
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from typing import Generator, Callable

## For convenience, use my credential set. IRL a company one would be used
plotly.tools.set_credentials_file(username='dischnerz', api_key='6TkYcsdEPZEH93e6RYU2')



##############################################################################
#                                   Globals
#----------*----------*----------*----------*----------*----------*----------*
log_level = logging.DEBUG
logging.basicConfig(stream=sys.stdout, level=log_level)
logger = logging.getLogger("Scheduler")

## Cool add color!
coloredlogs.install(fmt="(%(asctime)s) (%(levelname)s) [%(module)s.%(funcName)s #%(lineno)d]:  %(message)s")
coloredlogs.set_level(log_level)


_here = os.path.dirname(os.path.realpath(__file__))
_ACCESS_FILE = os.path.join(_here, "sample_ground_accesses.csv")
_WEIGHTS = {"utility": 1,
            "lost_imagery": -2,
            "eclipse_time":5,  # Not used, but included in case you wanted to use tiem in eclipse vs lost imagery as a metric
            "elevation":100}
_MIN_ANT_TRANSITION_TIME = 1 # minute
_MIN_SAT_CONTACTS = 3
_MAX_ANT_CONTACTS = 6
DF = pd.DataFrame()
##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*
class Access(dict):
    def __init__(self,access:dict):
        ## Different input types:
        if isinstance(access, pd.DataFrame):
            access = access.iloc[0]
        self.update(access)
        self._score = None


    @property
    def base_score(self):
        if self._score is None:
            self._score = compute_base_score(self)
        return self._score

    def __repr__(self):
        try:
            return f"Access: ({self['id']}) {self['satellite']} <--> {self['antenna']}" + \
                    f"({self['utility']:3.3f} MB), " + \
                    f"{self['start_time'].to_pydatetime()} - {self['end_time'].to_pydatetime()}, " + \
                    f"(unadjusted score: {self.base_score:3.3f} )"
        except:
            return "Access dictionary with unparseable parameters"
    
    def get_children(self, source_df:pd.DataFrame, search_limit:int=2):
        start_min_cutoff = self['end_time'] + relativedelta(minutes=5)
        start_max_cutoff = self['end_time'] + relativedelta(hours=search_limit)
        antenna_available = self['end_time'] + relativedelta(minutes=_MIN_ANT_TRANSITION_TIME)
        query = f"""( '{start_min_cutoff}' < start_time < '{start_max_cutoff}' ) 
                    and
                    (   (satellite=='{self['satellite']}' and start_time > '{self['end_time']}') 
                        or  
                        (satellite != '{self['satellite']}' and start_time > '{self['start_time']}') ) 
                     and 
                    (   (antenna != '{self['antenna']}') 
                        or 
                        (antenna == '{self['antenna']}' and start_time >= '{antenna_available}') 
                    )""".replace("\n"," ").replace("    ","")
        yield from ((Access(a)) for _, a in source_df.query(query).sort_values(by=['start_time','score'],ascending=False)[0:10].iterrows()) # Add a minute


class SchedulerGraph(object):
    def __repr__(self):
        return f"Graph with root at: {self.root['id']}. Best schedule from here: <{self.best_paths[0]}>"

    def __init__(self, source_file:str=_ACCESS_FILE, root_id:int=None, root_loc:int=None, weights:dict=_WEIGHTS, score_func:Callable=None):
        """Instantiate a SchedulerGraph object, which can build and analze a schedule possibilities graph in order to create a good schedule

        Kwargs:
            source_file:    CSV to find master schedule of access opportunities from
            root_id:        Optionally, specify the first access from which to build the graph by unique Access ID
            root_loc:       Optionally, specify the row of the access from which to build the graph
                    If neither are provided, one will be determined automatically
            weights:        Dictionary of weights to assign to various properties of the access. See _WEIGHTS for a template
            score_func:     Function that takes an access (key-value pairs described by a row in the `source_file`) and returns
                            a numeric score for a given access
        """
        self.df_master = load_accesses(source_file)
        self.weights = defaultdict(lambda:0)
        self.weights.update(weights)

        ## Adjustable priority maps per satellite and antenna
        self.ant_priority = defaultdict(lambda: 1.0)  # Default priority maps have no effect
        self.sat_priority = defaultdict(lambda: 1.0)

        ## Provided or default function for scoring an access, including adjusting multiplicatives for ant/sat prioritization
        if score_func is None:
            score_func = compute_base_score
        self.scorer = lambda access: score_func(access, weights=self.weights)*self.ant_priority[access["antenna"]]*self.sat_priority[access["satellite"]]

        ## Immediately rescore our accesses based on provided weights and whatnot
        self.rescore()

        ## Determine the root (starting place for the graph)
        self.root = None
        if root_id is not None:
            try:
                self.root = Access(self.df_master[self.df_master["id"] == root_id])
            except:
                logger.warn(f"Provided starting ID ({root_id}) does not exist in the access database ({source_file}). Trying to automatically determine starting Access")
        
        if self.root is None:
            if root_loc is not None:
                if root_loc <= len(self.df_master):
                    self.root = Access(self.df_master.iloc[root_loc])
                else:
                    logger.warn(f"Provided starting row ({root_loc}) does not exist in the access database ({source_file}). Trying to automatically determine starting Access")

        self.autoroot = False
        if self.root is None:
            self.autoroot = True
            self.pick_root()

        logger.info(f"Creating new SchedulerGraph() with starting access (root): {self.root}")

        ## Containers for the "optimal" schedule, as determined later
        self.optimal_schedule = None

        ## Some tracking variables for different graph searching algorithems
        self.paths = {}
        self.best_paths = [(self.scorer(self.root), (self.root["id"]))]
        self.nodes = []
        self.cache = {}

    def rescore(self):
        """Update 'score' column in the master 'database' of Accesses.
        """
        self.df_master["score"] = self.df_master.apply(lambda row: self.scorer(row), axis=1)

    def pick_root(self):
        """Pick the best root to start this graph from (best first Access to build the schedule upon)

        Best root is just the highest scoring access that occurs within the first two hours
        """
        query = f"start_time < '{self.df_master['start_time'].min() + relativedelta(hours=2)}'"
        self.root = Access(self.df_master.query(query).sort_values(by="score", ascending=False).iloc[0])
        return self.root

    def filter_accesses(self, ids:list)->pd.DataFrame:
        """Filter master schedule by a list of Access ids 

        Returns a view of the appropriate ids in the master schedule. Outside calling scope
        will need to perform copy() and reset_index() operations as needed
        """
        return self.df_master[self.df_master["id"].isin(list(ids))]

    def validate_schedule(self,ids:list, min_per_sat:int=3, max_per_ant:int=6, deconflict=False) -> (dict, dict, dict, dict):
        """Takes a list of ids that represent Accesses from the master schedule, and generate final outcome statistics

        Generates statistics (dictionaries), summarizing per satellite and antenna how much each was used by the 
        provided schedule. Will log warning messages if any usage is out of nominal bounds.
        """
        def check_for_overlap(df:pd.DataFrame, info:str="resource"):
            """Assumes dataframe is sorted by time, so adjacent Accesses (rows) whose start/end times overlap are in conflict
            """
            conflicts = []
            for ix in range(1,len(df)):
                if df.iloc[ix]["start_time"] < df.iloc[ix-1]["end_time"]:
                    logger.warn(f"Two scheduled accesses have overlapping {info} bookings! <{Access(df.iloc[ix-1])}> and <{Access(df.iloc[ix])}>")
                    conflicts.append(tuple([df.iloc[ix-1:ix+1]["id"]]))
            return conflicts

        logger.info(f"Validating schedule consisting of contacts: {ids}")
        df_sched = self.filter_accesses(ids)

        ## 1. See if each satellite got scheduled `min_per_sat` times
        sat_usage = {}
        sat_utility = {}
        conflicts = {}
        for sat in self.df_master["satellite"].unique():  ## Make sure we look at all satellites in the MASTER schedule
            df_sat = df_sched[df_sched["satellite"]==sat]
            sat_usage[sat] = len(df_sat)
            sat_utility[sat] = df_sat["utility"].sum()

            if sat_usage[sat] < min_per_sat:
                logger.warn(f"Satellite ({sat}) only had {sat_usage[sat]} contacts in this schedule! Minimum expected: {min_per_sat}")

            conflicts[sat] = check_for_overlap(df_sat, info="satellite")

        ## 2. See if Antennas got overbooked, as per `max_per_ant` specifies
        ant_usage = {}
        for ant in self.df_master["antenna"].unique():
            df_ant = df_sched[df_sched["antenna"]==ant]
            ant_usage[ant] = len(df_ant)

            if ant_usage[ant] > max_per_ant:
                logger.warn(f"Antenna ({ant}) had {ant_usage[ant]} contacts in this schedule! Maximum allowed: {max_per_ant}")

            conflicts[ant] = check_for_overlap(df_ant, info="antenna")
        return sat_usage, sat_utility, ant_usage, conflicts

    def generate_possibilities(self,path:list, hours_to_search:int=2, max_possibilities:int=10, min_hours_between_passes:int=4)-> Generator[Access, None, None]:
        """

        Kinda tricksy. Based on the `path` of Accesses already in the schedule, generates a 
        list of all viable 'next Access' possibilities, with some sconstraints. This function
        has to look at every satellite and antenna already booked, and make sure that the 
        returned possibilities do not overlap with any of them.

        Logic:
            -Find all accesses whose satellite/antennas overlap with exsting accesses in the `path`
            -Exclude those
            -Exclude anything that is to far down the timeline (configurable)
            -Exclude any negative scores (can happen based on instantiated _WEIGHTS)
            -Only take the top `max_possibilities` scores


        """
        schedule_sofar = self.filter_accesses(path)
        start_min_cutoff = schedule_sofar['end_time'].max() + relativedelta(minutes=5)
        start_max_cutoff = schedule_sofar['end_time'].max() + relativedelta(hours=hours_to_search)


        # antenna_available = self['end_time'] + relativedelta(minutes=_MIN_ANT_TRANSITION_TIME)

        query = f"( '{start_min_cutoff}' < start_time < '{start_max_cutoff}' )  and (score > 0)"

        satellite_avoid = []
        for sat in self.df_master["satellite"].unique():
            if sat in schedule_sofar["satellite"].unique():
                last_pass_end = schedule_sofar[schedule_sofar["satellite"] == sat]["end_time"].max()
            else:
                last_pass_end = schedule_sofar["start_time"].min()
            satellite_avoid.append(f"(satellite=='{sat}' and start_time <= '{last_pass_end + relativedelta(hours=min_hours_between_passes)}') ")

        antenna_avoid = []
        for ant in self.df_master["antenna"].unique():
            if len(schedule_sofar[schedule_sofar["antenna"]==ant]) >= _MAX_ANT_CONTACTS:
                antenna_avoid.append(f"(antenna=='{ant}' )")
            else:
                if ant in schedule_sofar["antenna"].unique():
                    last_pass_end = schedule_sofar[schedule_sofar["antenna"] == ant]["end_time"].max() + relativedelta(minutes=_MIN_ANT_TRANSITION_TIME)
                else:
                    last_pass_end = schedule_sofar["start_time"].min()
            antenna_avoid.append(f"(antenna=='{ant}' and start_time <= '{last_pass_end}') ")

        conflict_query = " and ".join(satellite_avoid) + " and " + " and ".join(antenna_avoid)
        avoid_df = self.df_master.query(conflict_query)
        possible_df = self.df_master[~self.df_master["id"].isin(avoid_df["id"])].query(query)
        yield from ((Access(a)) for _, a in possible_df.sort_values(by=['start_time'], ascending=True)[0:max_possibilities].iterrows()) # Add a minute


    def dfs_path(self, node=None, path=None, pathscore=0, store_all_paths=False, depth=0):
        """Depth-First-Search of access graph posibilities, where final score is calculated 
        whenever a leaf is reached.

        This is the 'truest' traversal of the graph, and gives much more power when it comes 
        to determining schedule validity *as it is is built*, and not after the fact. 

        Main detriment is that it is godawfully slow, since it truly traverses N! path combinations.
        Only way this is tennable for more than 100 nodes is if I figured out how to parallelize the 
        traversal. Not impossible but nontrivial nonetheless. 

        TODO:
            * Add stopping conditions for when we break ground station overbooking constraints
            * Check at a leaf whether or not all satellites have met their minimum downlink constraint
        """
        ## Initial conditions
        if path is None:
            path = tuple()
        if node is None:
            node = self.root

        newpath = path + tuple([node['id']])
        # Score of the path traversed from TOP down to (and including) this point, including adjustments
        pathscore += self.scorer(node)

        print("\rPath-Down DFS searching " + "."*depth, end="")
        if depth > 5:
            print("     (This could take a Loooooooooooooooooooooong time...)", end="")

        nochildren = True
        ## Explore children
        for child in self.generate_possibilities(newpath): #node.get_children():
            nochildren = False
            self.dfs_path(node=child, path=newpath, pathscore=pathscore, depth=depth+1)
        
        if store_all_paths:
            self.paths[path] = (pathscore, newpath)

        ## If we're at a leaf, store the top-down path and score if it is best
        if nochildren:  # `nochildren` means this is a leaf
            if pathscore > self.best_paths[0][0]:
                print(f"New best score found! {pathscore}")
                logger.debug(f"Schedule has new best score! {pathscore} Schedule: {path}")
                self.best_paths.insert(0,(pathscore,newpath))
        if node == self.root:
            print()

    def dfs_cached(self, node:Access=None, path:tuple=None, depth:int=0) -> (int,tuple):
        """Perform a depth-first-search of the graph, caching the best downward route as you go

        Recursive function. Children at each node are generated based on the path traversed 
        down to the current `node`. Each child is explored, and the best total score of the
        path traversed down from here (`node`->child->child...) is stored in the cache for 
        faster lookups next time around

        + This makes exploring a huge graph WAY faster. 
        - Caching best downward path means that the overall path chosen probably isn't perfect

        Todo: 

        """
        ## Initial conditions
        if path is None:
            path = tuple()
        if node is None:
            node = self.root

        print("\rCached DFS searching " + "."*depth, end="")
        nochildren = True
        possibilities = []
        newpath = path + tuple([node["id"]])
        # for child in node.get_children():
        for child in self.generate_possibilities(newpath):
            nochildren = False
            if child['id'] in self.cache:
                possibilities.append(self.cache[child['id']])
            else:
                possibilities.append(self.dfs_cached(node=child, depth=depth+1, path=newpath))
        
        ## Get the best possible path and score from here down
        if possibilities:
            ## Maybe filter out conflicted possibilities below?
            best_down_score, best_down_path = sorted(possibilities, key=lambda p:p[0], reverse=True)[0]
            best_score = self.scorer(node) + best_down_score  
            best_path = tuple([node["id"]]) + best_down_path
        else:
            best_score, best_path = self.scorer(node), tuple([node["id"]])

        if best_score>self.best_paths[0][0]:
            self.best_paths.insert(0,(best_score,best_path))

        self.cache[node['id']] = (best_score, best_path)
        if node == self.root:
            print()
        return best_score, best_path # Tuple of (score, (pathlist))

    def optimize_multipass(self, num_passes:int=3, ant_adj_rate:float=1.05, sat_adj_rate:float=1.25, publish:bool=False, deconflict:bool=False) -> (list,list,list):
        """Perform multiple passes of cached DFS scheduling search

        In between each run, self-adjust (har har har) the adjustment maps to try and boost
        chances of scheduling low performers. Akin to a classic 'squeaky wheel' scheduling algorithm.

        Kwargs: (Tuning parameters)
            num_passes:     Number of runs and score adjustments to make
            ant_adj_rate:   Divisor rate to adjust antenna priority which have too many bookings
            sat_adj_rate:   Multiplicitace rate to adjust satellites which do not get enough passes
            publish:        Create output products (see ScheduleGraph.publish())  

        Returns:
            scores:     List of overall schedule scores after each run (last one should be optimal)
            paths:      List of Access IDs (AKA a schedule) for each run
            stats:      List of summary statistics for each run     
        """
        logger.info(f"Running {num_passes} passes of DFS schedule optimization")
        scores = []
        paths = []
        stats = []

        ## For each schedule run:
        for ix in range(num_passes):
            self.cache = {} # Reset the cache to begin a fresh search
            
            ## Find the approximate best schedule in the Access Graph
            score, path = self.dfs_cached() 
            logger.info(f"Run {ix+1}: Best overall score {score}")
            ## Analyze performance
            sat_usage, sat_utility, ant_usage, conflicts = self.validate_schedule(path)

            ## Adjust adjustment maps
            for sat in sat_usage:
                if sat_usage[sat] < _MIN_SAT_CONTACTS:
                    logger.info(f"Satellite {sat} did not get enough passes ({sat_usage[sat]}). Applying higher weight for future runs: {self.sat_priority[sat]} -> {self.sat_priority[sat] * sat_adj_rate}")
                    self.sat_priority[sat] *= sat_adj_rate

            for ant in ant_usage:
                if ant_usage[ant] > _MAX_ANT_CONTACTS:
                    logger.info(f"Antenna {ant} had too many passes ({ant_usage[ant]}). Applying lower weight for future runs: {self.ant_priority[ant]} -> {self.ant_priority[ant]/(ant_adj_rate*ant_usage[ant]//6)}")
                    self.ant_priority[ant] /= (ant_adj_rate*ant_usage[ant]//6)
                ## If we're not using any contacts, bump that one espeecially
                if ant_usage[ant] < 3:
                    logger.info(f"Antenna {ant} just ({ant_usage[ant]}). Applying higher weight for future runs: {self.ant_priority[ant]} -> {self.ant_priority[ant]*ant_adj_rate}")
                    self.ant_priority[ant] *= ant_adj_rate

            ## Rescore accesses based on these new updates
            self.rescore()

            scores.append(score)
            paths.append(path)
            stats.append(tuple([sat_usage, sat_utility, ant_usage, conflicts]))

            ## Pick a new root if allowed
            if self.autoroot:
                self.pick_root()

            self.optimal_schedule = paths[-1]

        if deconflict:
            self.optimal_schedule = self.deconflict(self.optimal_schedule)

        if publish:
            logger.info("Publishing last run and writing sumamry reports")
            self.publish_sched()
        return scores, paths, stats

    def deconflict(self, ids:list) -> list:
        """Deconflict a schedule by dropping lowest scoring acceptable overages when a conflict arises
        """
        idset = set(ids)
        ###### 1. Drop accesses when too many bookings on an antenna
        sched_df = self.filter_accesses(idset).copy()
        for ant, df in sched_df.groupby("antenna"):
            if len(df) > _MAX_ANT_CONTACTS:
                logger.warn(f"Antenna {ant} has {len(df)-_MAX_ANT_CONTACTS} too many contacts! Dropping unnecessary contacts")
                sat_priority = sched_df["satellite"].value_counts()
                for sat, num_passes in sat_priority.iteritems():
                    if num_passes > _MIN_SAT_CONTACTS:
                        can_drop = df.query(f"satellite == '{sat}'")[0:min([len(df) - _MAX_ANT_CONTACTS, num_passes-_MIN_SAT_CONTACTS]):]
                        if len(can_drop) > 1:
                            logger.debug(f"Dropping {len(can_drop)} accesses for satellite {sat} for antenna {ant}")
                            idset = idset - set(can_drop["id"].values)
                            return self.deconflict(idset)
                        else:
                            continue
        return idset



    def publish_sched(self,ids:list=None) -> (str,str):
        """Publish schedule as a CSV, include summary statistic report

        Creates a directory to store files in with the timestamp
        """
        if ids is None:
            if self.optimal_schedule is None:
                logger.warn("Must provide an interable set of Access IDs, or run `SchedulerGraph.optmimize_multipass()` in order to publish anything")
                return
            ids = self.optimal_schedule

        logger.info(f"Publishing {len(ids)} Accesses as a new schedule, with associated reports included")
        df_sched = self.filter_accesses(ids).copy().reset_index(drop='index')
        df_sched["score"] = df_sched.apply(lambda row: self.scorer(row), axis=1)
        sched_score = df_sched["score"].sum()

        dirname = f"schedule_span{df_sched['start_time'].min().strftime('%Y%m%d')}-{df_sched['start_time'].max().strftime('%Y%m%d')}"
        if os.path.exists(dirname) is False:
            os.mkdir(dirname)

        ## Make a CSV file of the schedule itself
        csv_file = os.path.join(dirname,f"access_schedule-{pd.datetime.now().strftime('%Y%m%d-%H%M')}_score-{sched_score:.0f}.csv")
        df_sched.to_csv(csv_file, index=False)

        ## Write out summary statistics to a json file
        sat_usage, sat_utility, ant_usage, conflicts = self.validate_schedule(ids)
        sat_utility["total"] = sum(list(sat_utility.values()))
        report_file = os.path.join(dirname,f"ScheduleReport-{pd.datetime.now().strftime('%Y%m%d-%H%M')}_score-{sched_score:.0f}.json")
        with open(report_file, "w") as report:
            json.dump({"SatelliteUsage":sat_usage, 
                        "SatelliteUtility":sat_utility, 
                        "AntennaUsage":ant_usage, 
                        "ConflictList":conflicts}, 
                    report, indent=2)

        ## Create a static web page that redirects you to an online GANTT chart
        plotly_url = plot_schedule(df_sched)
        if plotly_url:
            with open(os.path.join(dirname, f"SchedulePlot_{pd.datetime.now().strftime('%Y%m%d-%H%M')}_score-{sched_score:.0f}.html"), "w") as plotpage:
                plotpage.write(f'<meta http-equiv="refresh" content="0; url={plotly_url.resource}" />')
            logger.info(f"Published schedule and associated reports/plots are now available in {dirname}")
        else:
            logger.info("No GANTT chart created, Try `SchedulerGraph().publish_sched()` again when you have an internet connection")
        
        logger.info(f"New schedule saved to {csv_file}, report statistics saved to: {report_file}")
        return csv_file, report_file

        

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
def compute_base_score(access:dict, weights:dict=_WEIGHTS):
    """Compute base score of an access according to weight dictionary

    Default module level _WEIGHTS dictionary will be used, but you can provide
    your own with whichever parameters you would like (smart defaultdict()) will
    not make use of any weight variables that aren't included)
    """
    # Add error checking, etc
    _weights = defaultdict(lambda:0.0)
    _weights.update(weights)
    return weights["utility"]*access["utility"] + \
            weights["lost_imagery"]*access["lost_imagery"] + \
            weights["elevation"]*pd.np.sin(access["max_el"]*pd.np.pi/180)
            # weights["eclipse_time"]*access["time_in_eclipse"] + \   # Other possible factor, included as example


def load_accesses(fname:str=_ACCESS_FILE) -> pd.DataFrame:
    """Loads accesses stored as a CSV into a dataframe
    
    CSV must have the same structure described in __doc__
    """

    def time_in_eclipse(access:pd.Series)->float:
        """
        Calculate time in eclipse, in seconds
        """
        if access["eclipse_state_end"] == access["eclipse_state_start"] == True:
            return access["duration"]

        if access["eclipse_state_end"] == access["eclipse_state_start"] == False:
            return 0

        if access["eclipse_state_start"] is True:
            return (access["terminator_time"] - access["start_time"]).total_seconds()

        return (access["end_time"] - access["terminator_time"]).total_seconds()

    ## Load the accesses into a dataframe 
    df = pd.read_csv(fname,index_col=0, converters={"end_time":pd.Timestamp, "start_time":pd.Timestamp, "terminator_time":pd.Timestamp, "midpoint":pd.Timestamp})

    ## Parse the `eclipse_state` from a string "[True, False]" into two separate bool columns
    df["eclipse_state_start"] = df.apply(lambda row: eval(row["eclipse_state"])[0], axis=1)
    df["eclipse_state_end"] = df.apply(lambda row: eval(row["eclipse_state"])[1], axis=1)
    df.drop("eclipse_state", axis=1, inplace=True)

    ## Add calculated time_in_eclipse to dataframe
    df["time_in_eclipse"] = df.apply(lambda row: time_in_eclipse(row), axis=1)

    ## Add the base score to the dataframe, makes indexing and sorting on score easier
    df['score'] = df.apply(lambda row: compute_base_score(row), axis=1)
    return df

DF = load_accesses()

def peek(df:pd.DataFrame):
    """Simple dev vocused function to look at handy subset of a big schedule dataframe
    """ 
    return df[["satellite","antenna","start_time","end_time","id","score"]]

def get_access(df:pd.DataFrame, accessid:int):
    """Return an Access by it's Unique ID, should be contained within the supplied `df` schedule
    """
    return Access(df.query(f"id=={accessid}").iloc[0])

def score_sched(ids:list):
    """Calculate schedule score identified by a list of `ids`

    IDs are assumed to come from master schedule
    """
    return filter_accesses(DF,ids)["score"].sum()


def plot_schedule(schedule_df):
    # https://plot.ly/python/gantt/
    def random_color():
        levels = range(32,256,32)
        return tuple(random.choice(levels) for _ in range(3))
    dfp = pd.DataFrame()
    tmp = []
    for _,access in schedule_df.iterrows():
        tmp.append({"Task":access["antenna"].replace("ground_site","ant-"), "Start":access["start_time"], "Finish":access["end_time"], "Resource":access["satellite"]})
    colors = {sat: f"rgb{random_color()}" for sat in schedule_df["satellite"].unique() }
    try:
        fig = ff.create_gantt(tmp, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
        url = py.iplot(fig, filename='Schedule', world_readable=True)
    except:
        logger.warning("Plotly services could not be reached, cannot create GANTT chart visual. Try logging again later")
        url = None 

    return url

##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__=="__main__":
    print("Demo")
    g = SchedulerGraph()
    _ = g.optimize_multipass()
    sys.exit(0)







