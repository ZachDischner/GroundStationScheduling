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
put on by Planet Labs. See prompt.md for details. 

Accesses should be stored as a csv file or equivalent, with the following properties per access

    ```
    ,antenna,eclipse_state,end_time,id,max_el,satellite,start_time,terminator_time,lost_imagery,midpoint,duration,utility
    ```

Algorithm:

Notes:

Examples:

"""

##############################################################################
#                                   Imports
#----------*----------*----------*----------*----------*----------*----------*
import os
import sys
import logging
import pandas as pd
import random
import arrow                        # Awesome time library
import plotly.plotly as py
import plotly
import plotly.figure_factory as ff
import coloredlogs
from collections import defaultdict
from dateutil.relativedelta import relativedelta

plotly.tools.set_credentials_file(username='DemoAccount', api_key='lr1c37zw81')



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
            "lost_imagery": -2}

##############################################################################
#                                   Classes
#----------*----------*----------*----------*----------*----------*----------*
class Node(object):
    def __repr__(self):
        return f"Access {self.access['id']} - Best: {self.bestscore}"
    def __init__(self, access):
        self.access = access
        self.children = []
        self.bestscore = None  # --> (score,[path])

    def get_children(self):
        start_cutoff = self.access['end_time'] + relativedelta(hours=2)
        query = f"""( start_time < '{start_cutoff}' ) 
                    and
                    (   (satellite=='{self.access['satellite']}' and start_time > '{self.access['end_time']}') 
                        or  
                        (satellite != '{self.access['satellite']}' and start_time > '{self.access['start_time']}') ) 
                     and 
                    (   (antenna != '{self.access['antenna']}') 
                        or 
                            (antenna == '{self.access['antenna']}' 
                            and 
                            start_time > '{self.access['end_time']}') 
                    )""".replace("\n"," ").replace("    ","")
        yield from (Node(Access(a)) for _, a in df.query(query).sort_values(by='score')[0:10].iterrows()) # Add a minute

cache = {}

class Graph(object):
    def __repr__(self):
        return f"Graph with root at: {self.root.access['id']}. Best schedule from here: <{self.best_paths[0]}>"
    def __init__(self, root):
        self.root = root
        self.paths = {}
        self.best_paths = [(root.access.score, (root.access["id"]))]
        self.nodes = []

    def dfs(self, node=None, path=None, pathscore=0):
        if path is None:
            path = tuple()
        if node is None:
            node = self.root
        pathscore += node.access.score  # Score of the path traversed from TOP down till this point
        # print(f"Depth: {len(path)} - Score: {pathscore:3.5f}")

        nochildren = True
        possibilities = []
        for child in node.get_children():
            nochildren = False
            # self.nodes.append(child)
            if child.access['id'] in cache:
                possibilities.append(cache[child.access['id']])
            else:
                newpath = path + tuple([child.access['id']])
                possibilities.append(self.dfs(node=child, path=newpath, pathscore=pathscore))
        
        ## Get the best possible path and score from here down
        if possibilities:
            best_down_score, best_down_path = sorted(possibilities, key=lambda p:p[0], reverse=True)[0]
            best_score = node.access.score + best_down_score  
            # best_path = path + best_down_path   # No double counting...
            best_path = tuple([node.access["id"]]) + best_down_path
        else:
            # best_score, best_path = pathscore, path
            best_score, best_path = node.access.score, tuple([node.access["id"]])

        # if nochildren:
            # self.paths[path] = pathscore
        if best_score>self.best_paths[0][0]:
            print(f"Schedule {path} has the new best total score: {pathscore}")
            self.best_paths.insert(0,(best_score,best_path))

        print(f"At node ({node}), best score, path = ({best_score}) ,({best_path})")
        cache[node.access['id']] = (best_score, best_path)
        return best_score, best_path # Tuple of (score, (pathlist))
        
        # node.bestscore = (pathscore,path)

class Scheduler(object):
    def __init__(self, access_report:str=_ACCESS_FILE, span:int=10):
        logger.info(f"Creating new Scheduler for access in {access_report}")
        self.df = load_accesses(access_report)
        logger.info(f"Loaded {len(self.df)} accesses for analysis and scheduling")

        ## Final Schedules
        self.schedules = defaultdict(list)

        ## Initialize pre-schedule variables
        self.span = span
        self.initialize()

    def initialize(self):
        """Initialize per-schedule run variables
        """
        ## Dictionaries to track recent satellite and ground station usage
        self.recent_downlinks = {}
        self.gs_usage = {}

        ## Track currently active satellites and ground stations
        self.sat_active = {}
        self.ant_active = {}

        ## Initial conditions for building our schedule
        self.span_start = arrow.get(min(self.df["start_time"])).replace(seconds=-1)

    def get_accesses(self):
        ## Get all possible access between `span_start` and `span_start + span` minutes
        t1 = self.span_start
        t2 = self.span_start.replace(minutes=+self.span)
        subset_df = self.df.query(f"start_time >= '{t1}' and end_time <= '{t2}'")
        logger.debug(f"Initially found {len(subset_df)} accesses to consider for time span: {t1} - {t2}")
        ## Filter out any satellites or Resources that are currently in use
        #    Potential access:                  [_____________satX_____________]
        #    Currently active:  ...________________]
        # Drop that access from consideration since it overlaps with an existing active downlink
        ix_to_drop = []
        for sat in self.sat_active:
            ix_to_drop.append((subset_df.query(f"satellite=='{sat}' and start_time < '{self.sat_active[sat]['end']}'").index))
        logger.debug(f"Found {len(ix_to_drop)} accesses that overlap with existing satellite bookings")
        subset_df.drop(subset_df.index[ix_to_drop])

        ## Same thing for antenna
        ix_to_drop = []
        for antenna in self.ant_active:
            ix_to_drop.append((subset_df.query(f"antenna=='{antenna}' and start_time < '{self.gs_active[antenna]['end']}'").index))
        logger.debug(f"Found {len(ix_to_drop)} accesses that overlap with existing antenna bookings")
        subset_df.drop(subset_df.index[ix_to_drop])

        logger.debug(f"After filtering, found {len(subset_df)} accesses to consider for time span: {t1} - {t2}")

        return [Access(access) for _,access in subset_df.iterrows()]

    def base_score(self, access_list):
        pass



class Access(dict):

    def __init__(self,access:dict):
        self.update(access)

    @property
    def score(self):
        return self.compute_score()

    def __repr__(self):
        try:
            return f"Access: ({self['id']}) {self['satellite']} <--> {self['antenna']}" + \
                    f"({self['utility']:3.3f} MB), " + \
                    f"{arrow.get(self['start_time']).format('YYYY-MM-DD HH:mm:ss')} - {arrow.get(self['end_time']).format('YYYY-MM-DD HH:mm:ss')}, " + \
                    f"(score: {self.score:3.3f})"
        except:
            return "Access dictionary with unparseable parameters"

    def compute_score(self):
        return _WEIGHTS["utility"]*self["utility"] + _WEIGHTS["lost_imagery"]*self["lost_imagery"]

##############################################################################
#                                   Functions
#----------*----------*----------*----------*----------*----------*----------*
# Tired of looking at big dataframes
def peek(df):
    return df[["satellite","antenna","start_time","end_time","id","score"]]

def normalize(x):
    return x/((max(x)-min(x)))

def get_access(accessid):
    return Access(df.query(f"id=={accessid}").iloc[0])

def score_sched(ids):
    return filter_accesses(df,ids)["score"].sum()

def filter_accesses(df,ids):
    return df[df["id"].isin(ids)]


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

    ## Also normalize, not sure if we want to or not. I think so
    for col in ["lost_imagery", "duration", "utility", "time_in_eclipse"]:
        df["N"+col] = normalize(df[col])

    return df

def plot_schedule(schedule_df):
    # https://plot.ly/python/gantt/
    def random_color():
        levels = range(32,256,32)
        return tuple(random.choice(levels) for _ in range(3))
    dfp = pd.DataFrame()
    tmp = []
    for _,access in schedule_df.iterrows():
        tmp.append({"Task":access["antenna"], "Start":access["start_time"], "Finish":access["end_time"], "Resource":access["satellite"]})
    colors = {sat: f"rgb{random_color()}" for sat in schedule_df["satellite"].unique() }
    return tmp,colors
    fig = ff.create_gantt(tmp, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    py.iplot(fig, filename='Schedule', world_readable=True)



df = load_accesses()
df['score'] = df.apply(lambda row: Access(row).score, axis=1)
##############################################################################
#                              Runtime Execution
#----------*----------*----------*----------*----------*----------*----------*
if __name__=="__main__":
    pass







