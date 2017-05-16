Prompt - Ground Station Scheduling
================================================


# The problem - Ground station scheduling
Planet operates a large fleet of satellites with a large network of ground stations.  One aspect of the scheduling problem is to ensure that all the satellites get allocated enough ground station time to download the imagery.  The task is to assign ground station contacts for a constellation of 12 satellites to a network of 16 ground stations in 24 hour period.
 
## Things to include:
- Each satellite must have a minimum of 3 ground contacts per day.
- One ground station can only talk to one satellite at the same time, and vice-versa.
- Each ground station must be allowed a minimum 1 minute transition down time between subsequent contacts.
- Each ground station can support a maximum of 6 contacts per day to prevent wear and tear.
 
## Other considerations:
- We want to maximize the amount of data we can get down (for each sat and the constellation).
- Ground contacts in eclipse are preferred as we cannot image and downlink simultaneously.
- For each satellite 3 contacts spread 8 hrs apart is better than 3 contacts spread 30 min apart.
 
Attached is a csv containing everything you need for the problem.  The header description is as follows;
 
* id = A unique ID number of the access
* satellite = Name of the satellite. 
* antenna = Name of the antenna. Antennas with the same site name are co-located ie ground_site1_2a and ground_site1_2b are co-located.
* max_el = Maximum elevation of the access (in degrees)
* start_time = Start time of the access
* midpoint = Time of mid point of the access
* end_time = End time of the access
* duration = Total duration of the access (seconds)
* eclipse_state = Boolean describing the eclipse state at the start and end of the access i.e. [True, True] is total eclipse
* terminator_time = The time the satellite crosses the terminator, if applicable i.e. goes from sunlight to darkness 
* utility = The amount of data (in MB) that can be downloaded in the access
* lost_imagery = The amount of imagery (in MB) that cannot be taken if we choose this access.  We cannot image and downlink simultaneously. For eclipse accesses this is zero as we cannot image at night.
 
## Your solution
- Please feel free to use as many or as few of the variables as you want.
- The choice of programming language and tools is up to you (python, c++ is preferred).  
- Please think about the presentation of your solution and code structure. According to  Akin's law number 20 "A bad design with a good presentation is doomed eventually. A good design with a bad presentation is doomed immediately."
- Submitted code should be able to compile/run and reproduce the same results.
- We are looking for creativity, innovation and elegance in the solution, not perfection.  Submit your best solution, even if it doesn't completely satisfy the above conditions.
 
NOTE: Please do not share this problem or data with anyone outside of Planet.
Please submit here: https://app.greenhouse.io/tests/56f0749c57df9cdf3a0d439299ef64d7