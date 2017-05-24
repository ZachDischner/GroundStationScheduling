Ground Station Optimizer
=============================================

Zach Dischner
May 24 2017

## Intro and Install
This module attempts to generate a valid satelltie downlink schedule from a set of Accesses. Examples of the structure of an Access can be seen in `sample_ground_accesses.csv`.

```
    ,antenna,eclipse_state,end_time,id,max_el,satellite,start_time,terminator_time,lost_imagery,midpoint,duration,utility
```

The big idea is to score each access based on three criteria, and build a valid schedule based an assembly of said scores:  

* Utility - Amount of data downlinked (+)
* Lost Imagery - Opportunity cost of the downlink (-)
* Elevation - Highest elevation that an access achieves wrt ground antenna (+)

### Python
Anaconda's Python distribution is recommended for getting started. Easiest way to get started is to install and create a new Python 3.6 environment 
from the included `environment.yml` file. http://conda.pydata.org/docs/using/envs.html.

```
conda env create -f environment.yaml
source activate schedulerpy 
```

Otherwise, if you don't have Anaconda, a working Python 3.6+ environment with a few ancilarly modules is all you need. 
Consult `environment.yml` for a list of all modules and install however you're most comfortable.

### Clone

```
git clone git@github.com:ZachDischner/GroundStationScheduling.py
```

### Run
Example scheduling run is found in the top level `SchedulerDemo.ipynb`

## Algorithm 
The big idea here is to generate a schedule by creating a possibility graph of accesses, and finding the highest scoring path through the graph. 

#### Scheduler Graph
`scheduler.py` will build a graph, where each graph node is represented by an access, and a node's children are represented by possible future accesses, where `possible` is computed based on the availability of satellties and resources based on what is in use by the current node.

The graph is constructed on the fly as it is explored. To start, just provide the path to the accesses csv.

```python
import scheduler

graph = scheduler.SchedulerGraph(source_file="sample_ground_accesses.csv")
graph.root

#In [448]: graph.root  # Chooses root automatically
#Out[448]: Access: (7181005) sat1 <--> ground_site1_2a(9031.069 MB), 2017-02-15 23:53:36.101531 - 2017-02-16 00:00:34.101531, (unadjusted score: 8357.217 )
```

![Graph](http://i.imgur.com/NGFYaf7.jpg)


#### Graph Searching
The best path through a graph is defined as the path from the root to a leaf with the highest cumulative score. The cumulative score is the summed raw score of each node, with per-satellite and per-antenna prioritization adjustments applied. This path (which is just a list of nodes, which is just a list of access IDS) makes up the downlink schedule

**True DFS**

Actually searching the entire possibility graph for the optimal schedule takes untenably long. While this is the *correct* way to find a schedule, it just takes far too long as is. 
    
    * Future work here - heavy parallelization of the search would make this option feasable

You can perform this search, which will score the best found path/schedule as the graph is searched.

```python
# Start by considering only the last 200 possible accesses
smallgraph = scheduler.SchedulerGraph(root_loc=-200) 
smallgraph.dfs_path()
smallgraph.best_paths[0] # Best schedule, stored as a tuple (score, (id_list))
```

**Cached DFS**

In order to sacrafice some correctness for speed, the `SchedulerGraph` has a method to search the graph and cache results along the way. At every node, the graph will store the score of the current node + the best path from the node downward. As such, the next time the node is explored, the graph already knows the best route from said node, through each of its children, and down to the graph leaves. This allows the graph to be fully explored in a reasonable timeframe. 

This introduces possible errors due to the fact that the path taken to a given node and the best (cached) path downward from that node might not be fully compatible. Typically, this manifests as overbookings of some antennas. This shortcoming is known and is just a part of this first-order approximation

```python
graph = scheduler.SchedulerGraph() 
best_score, best_path = smallgraph.dfs_cached()
```

### Optimal Schedule Finding
The preferred method of schedule finding is to use the `SchedulerGraph().optimize_multipass()` method. This will perform many iterations of the cached graph search. In between each run, `SchedulerGraph` will reprioritize satellties and antennas that are over/underbooked. Log messages will display these reweights along the way. 

```python
## Search and re-weight 10 times. Publish output report and GANTT chart when finished
scores, paths, stats=graph.optimize_multipass(publish=True,num_passes=10)

## Optimal schedule
graph.optimal_schedule   # Tuple of access IDs
```
**Deconfliction**

Due to the potential errors in cached path matching, the `SchedulerGraph()` object also contains a simple `deconflict` method, which will drop accesses from antennas with too many bookings, while attempting to keep the required minimum satellite contacts. To perform deconfliction after multipass schedule optimization before publishing results, call with the kwarg: `graph.optimize_multipass(deconflict=True)`

## Output Products
After determining a schedule, output products can be created via the `SchedulerGraph().publish_sched()` method. It creates a named folder containing:

* An output schedule CSV of accesses
* A JSON report with summary statistics including usage and conflicts
* A static HTML page which redirects you to a `plotly` interactive GANTT chart

**JSON Report Sample**

```json
{
  "SatelliteUsage": {
    "sat1": 8,
    "sat2": 3,
    ...
    "sat12": 4
  },
  "SatelliteUtility": {
    "sat1": 70162.11566313001,
    "sat2": 23822.80584263,
    ...
    "sat12": 37992.88309842,
    "total": 473714.23716567
  },
  "AntennaUsage": {
    "ground_site1_2a": 0,
    "ground_site1_2b": 15,
    ...

```

**GANTT Chart**

![GANTT](http://i.imgur.com/BupKaQE.png)

### Customizeability
See the code documentation for more details, but you can customize the scoring function, weights, and optimization settings to suit your needs. EG

```python
# Use your own scoring function
def random_scorer(access):
    return random.randint(100)
    
graph = scheduler.SchedulerGraph(scorer=random_scorer)
...
```

## Future Work
This first order approximation takes a number of shortcuts, and as such, suffers from a few pitfalls. Immediate future explorations include (in order of priority):

* Attempt to formulate this as an **Integer (boolean) Optimization Problem**
    * https://developers.google.com/optimization/mip/integer_opt
* Parallelize true DFS search to avoid choosing a schedule with conflicts
* Attempt to formulate as a **Vehicle Delivery Optimization Problem** with specific time windows 
    * https://developers.google.com/optimization/routing/tsp/vehicle_routing_time_windows 
* Utilize Google's more powerful gantt plotting library












