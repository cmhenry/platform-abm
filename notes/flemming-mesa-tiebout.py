'''Nathaniel Flemming 28/2/23'''
#  Agent Based Modeling package imports
from mesa import Model, Agent  # model and agent classes
from mesa.time import BaseScheduler  # scheduler for agents to take steps
from mesa.space import MultiGrid  # type of grid agents will be placed on
from mesa.datacollection import DataCollector  # data collector to pull information once model is done running
import random  # for placing agents randomly
import numpy as np  # math
import pandas as pd  # data frames for data collector output


############## Create Resident agent class
class Resident(Agent):
    def __init__(self, id, model, preference):  # unique id, model, a preference
        super().__init__(id, model)  # super lets you take arguments from other classes, in this case model
        self.preference = preference  # set preference to the value that was passed
        self.current_city = None  # Resident's current city will be set later, but needs to be initialized here

    def step(self):
        neighbor_spending = []  # initialize list of neighboring cities' spending
        for neighbor in self.model.grid.iter_neighbors(self.pos, moore = True, include_center=True):  # iterate through all neighbors
            if isinstance(neighbor, City) and neighbor.pos == self.pos:  # if the neighbor is a city
                self.current_city = neighbor  # current city = neighbor city with same position as resident
                current_gap = abs(self.current_city.spending_level - self.preference)  # calculate gap between spending and preference
                self.model.gap += current_gap  # update model, add current spending gap to model tally
                # this is the gap between the agent's preference and the city's most recent spending level
                # the spending level may be different from when the agent first moved here since cities activate second
            if isinstance(neighbor, City):
                neighbor_spending.append(neighbor.spending_level)  # fill in list of cities' spending
        smallest_gap = min(neighbor_spending, key=lambda x: abs(x-self.preference))  # find smallest spending-preference gap
        if smallest_gap < current_gap:  # if there's a city with a smaller gap than the current one add it to a list
            candidates = [neighbor for neighbor in self.model.grid.iter_neighbors(self.pos, moore=True, include_center=True)
                          if isinstance(neighbor, City) and neighbor.spending_level == smallest_gap]
            if candidates:  # if there are any cities in the list...
                new_city = random.choice(candidates)  # one of the cities is chosen randomly
                self.model.grid.move_agent(self, new_city.pos)  # Resident moves to the new city
                self.current_city = new_city  # set current city to the new city


############ Create City agent class
class City(Agent):
    def __init__(self, id, model, spending_level):  # id, model, spending level
        super().__init__(id, model)
        self.spending_level = spending_level  # set spending level to value passed

    def step(self):  # City looks at residents in and around it and adjusts spending to match mean preference, if there are residents
        resident_preferences = []  # initialize resident preferences list
        for neighbor in self.model.grid.iter_neighbors(self.pos, moore = True, include_center=True):  # find preferences of all neighboring residents
            if isinstance(neighbor, Resident):  # If a neighbor is a resident...
                resident_preferences.append(neighbor.preference)  # append their preference to a list
        if resident_preferences:  # if resident preferences list isn't empty (i.e. there's at least 1 neighboring resident)...
            self.spending_level = np.mean(resident_preferences)  # calculate mean preference and set spending equal to it
        self.model.spending_levels.append(self.spending_level)  # add spending to model level list of spending levels


############  Create Model class
class MiniModel(Model):
    def __init__(self, num_residents, height, width, num_cities, init_spending_levels, preferences, min_gap):
        self.num_residents = num_residents  # desired number of residents
        self.height = height  # height of grid
        self.width = width   # width of grid
        self.num_cities = num_cities  # desired number of cities (currently one city per cell)
        self.init_spending_levels = init_spending_levels   # city spending levels
        self.preferences = preferences   # resident preferences
        self.min_gap = min_gap   # minimum total gap between spending and preferences that will make the model stop
        self.schedule = BaseScheduler(self)  # schedule for which Resident and city moves when, they activate in order
        self.grid = MultiGrid(width, height, torus=True)  # create grid, set torus so no edge
        self.gap = 0  # start at 0 spending-preference gap, will update when stepping
        self.spending_levels = []  # create empty spending levels list, will fill when stepping
        # data collector to pull information out of the model when it's done running
        self.datacollector = DataCollector({"gap": lambda m: m.gap, "spending_levels": lambda m: m.spending_levels})
        self.running = True  # whether model is still running

        ##  Create Resident agents
        #  Because residents are added to the schedule first, they will move first, since agents activate in order
        for i in range(self.num_residents):
            #  choose random grid coordinates
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            resident = Resident(i, self, self.preferences[i])  # create resident and assign id and preference
            self.grid.place_agent(resident, (x, y))  # place agent on grid at random location
            self.schedule.add(resident)  # add agent to schedule

        ##  Create City agents
        k = 0  # create new variable to increment
        id = i + 1  # create unique City ids starting where Residents leave off
        for cell in self.grid.coord_iter():
            # set place on grid, iterating through grid coordinates
            x = cell[1]  # cell's x coordinate
            y = cell[2]  # cell's y coordinate
            city = City(id, self, self.init_spending_levels[k])  # create city and assign id and spending
            self.grid.place_agent(city, (x, y))  # place agent on grid at next location
            self.schedule.add(city)  # add agent to schedule
            k += 1  # increment spending index
            id += 1  # increment city id

    ## model's step function
    def step(self):  # run model: has agent move, updates gap, checks if gap is small enough to stop
        self.gap = 0  # reset spending-preference gap
        self.schedule.step()   # agents all take a step
        self.datacollector.collect(self)  # collect data at each step, from instance of class
        if self.gap < self.min_gap:  # check if gap is greater than some set value
            self.running = False  # If the gap between spending and preferences is small enough, stop


############ Set Model Parameters
random.seed(123)  # set seed for reproducible randomness
np.random.seed(123)  # set seed for reproducible randomness
num_residents = 300  # desired number of residents
height = 10  # height of grid
width = 10  # width of grid
num_cities = height*width  # desired number of cities (currently one city per cell)
init_spending_levels = np.random.randint(1, 21, num_cities)  # city spending levels
preferences = np.random.randint(1,21, num_residents)  # resident preferences
min_gap = 5*num_residents  # minimum total gap between spending and preferences that will make the model stop
# create model
model = MiniModel(num_residents, height, width, num_cities, init_spending_levels, preferences, min_gap)


######## Run model and examine result
steps = 10  # max number of steps the model will take

## Run model for 20 steps
for step in range(steps):  # take 20 steps
    model.step()  # model/agents take a step
    model_out = model.datacollector.get_model_vars_dataframe()  # collect info from model
    print(model_out.gap)  # print the total gap at that step

model_out.gap.plot()  # plot the total gap over time

# spending levels is a series of identical lists, one for each step with every spending level that occurred
df = pd.DataFrame(model_out.spending_levels)  # extract spending levels from model output as a data frame
row1 = pd.DataFrame(df['spending_levels'].iloc[0])  # get just the first row, since all rows are identical
spending_matrix = np.reshape(row1.values, (model.schedule.steps, num_cities))  # reshape that row into a num of steps by num of cities array
spending_matrix = np.vstack([init_spending_levels, spending_matrix])  # add original spending preferences as first row of matrix
spending_df = pd.DataFrame(spending_matrix)  # convert back to data frame
spending_df.plot(legend=False, alpha=0.1, color='blue')  # use plotting to show spending levels over time
# calculate mean spending and variance in spending at each step
spending_summary = pd.DataFrame(zip(spending_df.mean(axis=1), spending_df.var(axis=1)))
spending_summary.columns = ['mean', 'variance']  # name columns

spending_summary['mean'].plot(color='black')  # plot mean spending
spending_summary['variance'].plot()  # plot variance in spending
#spending_summary['mean'][20]
np.mean(preferences)  # compare to mean resident preference

## Run model until the gap falls low enough or it reaches 20 steps
while model.running and model.schedule.steps < steps:  # run until gap falls below a threshold, or for N steps
    model.step()  # model/agents take a step
    model_out = model.datacollector.get_model_vars_dataframe()  # collect info from model
    print(model_out.gap)  # print the total gap at that step

print(model.schedule.steps)  # how many steps did the model take
model_out.gap.plot()  # plot gap over time

df = pd.DataFrame(model_out.spending_levels)  # extract spending levels from model output as a data frame
row1 = pd.DataFrame(df['spending_levels'].iloc[0])  # get just the first row, since all rows are identical
spending_matrix = np.reshape(row1.values, (model.schedule.steps, num_cities))  # reshape that row into a num of steps by num of cities array
spending_matrix = np.vstack([init_spending_levels, spending_matrix])  # add original spending preferences as first row of matrix
spending_df = pd.DataFrame(spending_matrix)  # convert back to data frame
spending_df.plot(legend=False, alpha=0.1, color='blue')  # use plotting to show spending levels over time
# calculate mean spending and variance in spending at each step
spending_summary = pd.DataFrame(zip(spending_df.mean(axis=1), spending_df.var(axis=1)))
spending_summary.columns = ['mean', 'variance']  # name columns

spending_summary['mean'].plot(color='black')  # plot mean spending
spending_summary['variance'].plot()  # plot variance in spending
#spending_summary['mean'][20]
np.mean(preferences)  # compare to mean resident preference