#!/usr/bin/env python
# coding: utf-8

# # exploring tiebout abm with python agentpy

# In[51]:


# Model design
import agentpy as ap
import networkx as nx
import numpy as np
import random

# Visualization
# import matplotlib.pyplot as plt
# import seaborn as sns
# import IPython

'''
TIEBOUT CYCLE
1. setup model: assign preferences, policies, communities to platforms
2. communities: calculate utility, set strategies, gather platform candidates
3. communities: relocate/stay
4. platforms: aggregate preferences and recalibrate policies
5. repeat 2-4 until communities exhaust strategies
'''

# to dos:
# platform preference aggregation scheme

class Community(ap.Agent):
    '''
    community characteristics:
        preferences (binary [0,1,...] or continuous range(parameter p_space))
        utility function -(community pref - platform policy)^2
    '''
    
    kind = 'community'

    def setup(self):
        """ Initialize a new variable at agent creation. """
        # set preferences
        if self.p.p_type == 'binary':
            self.preferences = [random.choice([0, 1]) for _ in range(self.p.p_space)]
        elif self.p.p_type == 'non-binary':
            self.preferences = random.randrange(self.p.p_space)
        # set initial vars
        self.current_utility = 0
        self.platform = ''
        self.strategy = ''

    def utility(self,platform):
        """ calculate utility over a given platform from utility fn """
        # basic utility function, no sub-game
        if self.p.p_type == 'binary':
            if len(self.preferences) != len(platform.policies):
                raise ValueError("agent must have complete preferences over vector of platform policies")
            utility = sum(pref == pol for pref,pol in zip(self.preferences, platform.policies))
        elif self.p.p_type == 'non-binary':
            utility = -np.square(self.preferences - platform.policies)
        return(utility)

    def update_utility(self):
        """ updated utility from current platform """
        self.current_utility = self.utility(self.platform)
    
    def join_platform(self, platform):
        """ join a platform """
        self.platform = platform
    
    def find_new_platform(self):
        """ find candidate platforms """
        candidates = []
        for platform in self.model.platforms:
            if self.utility(platform) > self.current_utility:
                candidates.append(platform)
        return(candidates)
    
    def set_strategy(self):
        """ compare utilities and pick new platform """
        # current_utility = self.update_utility(self.platform)
        for platform in self.model.platforms:
            if self.utility(platform) > self.current_utility:
                self.strategy = 'move'
                return
            else:
                self.strategy = 'stay'


class Platform(ap.Agent):
    '''
    platform characteristics:
        policies
        communities
        preference aggregation mechanisms
    '''
    
    kind = 'platform'
    
    def setup(self):
        """ initialize new variables at platform creation. """
        # set policies
        if self.p.p_type == 'binary':
            self.policies = [random.choice([0, 1]) for _ in range(self.p.p_space)]
        elif self.p.p_type == 'non-binary':
            self.policies = random.randrange(self.p.p_space)

        self.communities = []
        self.community_preferences = []
        self.ls_utilities = {}
    
    def add_community(self, community):
        """ add community to platform """
        self.communities.append(community)
        
    def rm_community(self, community):
        """ rm community from platform """
        self.communities.remove(community)

    def community_utilities(self):
        """ obtain utilities for attached communities """
        for community in self.communities:
            self.ls_utilities[community.id] = community.utility(self)
    
    def aggregate_preferences(self):
        """ build an array of community preferences """
        for community in self.communities:
            self.community_preferences.append(community.preferences)

    def direct_vote(self):
        """ survey communities and aggregate preferences with direct votes """

        self.aggregate_preferences()

        aggregated_votes = []
        num_policies = len(self.policies)
        num_communities = len(self.communities)

        for i in range(num_policies):
            votes = 0

            for j in range(num_communities):

                if self.policies[i] == self.community_preferences[j][i]:
                    votes += 1

            aggregated_votes.append(votes)

        return aggregated_votes

    def movement_formation(self):
        """ survey communities, construct movements, aggregate preferences """

        model.setup()
        testPlatform = model.platforms[0]
        testPlatform.aggregate_preferences()
        testPlatform.community_preferences

        testComm = testPlatform.communities[0]

        # generate random movements
        movement = [random.choice([0, 1]) for _ in range(model.p.p_space)]

        # compute fitness
        fitness = sum(pref == pol for pref,pol in zip(testComm.preferences, movement))

        # hill cimb
        
        


    def election(self):
        """ election mechanism """

        if(self.p.institution == 'direct'):
            ## gather votes
            votes = self.direct_vote()

            ## set threshold
            threshold = np.ceil(len(self.communities))
            num_policies = len(self.policies)

            ## count votes
            for i in range(num_policies):
                if self.votes[i] < threshold:
                    ## invert policy if it's below the threshold
                    self.policies[i] = self.polices[i] ^ 1

        if(self.p.institution == 'platform'):
            ## gather votes
            votes = self.indirect_vote()


        

class MiniTiebout(ap.Model):

    def setup(self):
        """ Initialize the agents and network of the model. """
        # prepare network
        # graph = nx.Graph()
        
        # add communities to network
        self.communities = ap.AgentList(self, self.p.n_comms, Community)
        # for community in self.communities:
        #     graph.add_node(community)
        
        # add platforms to network
        self.platforms = ap.AgentList(self, self.p.n_plats, Platform)
        # for platform in self.platforms:
        #     graph.add_node(platform)
        
        # randomly add communities to platforms
        for community in self.communities:
            platform = self.random.choice(self.platforms)
            community.join_platform(platform)
            platform.add_community(community)
            # graph.add_edge(community, platform)
        
        # combine into agentlist & register network
        self.agents = self.communities + self.platforms
        # self.network = self.agents.network = ap.Network(self, graph)
        # self.network.add_agents(self.agents, self.network.nodes)
    
    def satisfied(self):
        """ check state of communities for equlibrium """
        for community in self.communities:
            if community.strategy == 'move':
                return False
        return True

    def update(self):
        """ Record variables after setup and each step. """
        # update utilities for current platform-community relationships
        for community in self.communities:
            community.update_utility()
            community.set_strategy()

        # record community strategies, utilities, and platforms
        for platform in self.platforms:
            # average utility
            platform.community_utilities()
            avg_utility = sum(platform.ls_utilities.values()) / float(len(platform.ls_utilities))
            self.record(f'{"avg_util"}{platform.id}', 
                avg_utility)

        for community in self.communities:
            # location
            self.record(f'{"loc_com"}{community.id}', 
                community.platform)
            # strategy
            self.record(f'{"strat_com"}{community.id}',
                community.strategy)
            # utility
            self.record(f'{"util_com"}{community.id}',
                community.current_utility)
        
        # check if every community is happy
        if self.satisfied():
            self.stop()
                
    def step(self):
        """ Define the models' events per simulation step. """
        # search for new platforms
        for community in self.communities:
            if community.strategy == 'move':
                community.platform.rm_community(community)
                candidates = community.find_new_platform()
                platform = self.random.choice(candidates)
                community.join_platform(platform)
                platform.add_community(community)



parameters = {
    'n_comms': 1000,
    'n_plats': 100,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50
}

model = MiniTiebout(parameters)
model.setup()

ptest = model.platforms[0]
ptest.aggregate_preferences()

results = model.run()
results.variables.MiniTiebout

def stackplot(data, ax):
    """ stackplot of average utility by platform """
    

def animation_plot(model, axs):
    axs.set_title("average utility")

    stackplot()

fig, axs = plt.subplots(1, figsize=(8,8))
animation = ap.animate(MiniTiebout(parameters), fig, axs, animation_plot)

T = np.array([
  [[1,2,3],    [4,5,6],    [7,8,9]],
  [[11,12,13], [14,15,16], [17,18,19]],
  [[21,22,23], [24,25,26], [27,28,29]],
  ])
print(T.shape)
print(T)

# model.setup()
# model.update()
# report(model)
# model.step()
# report(model)
# model.update()
# report(model)

# def report(model):
#     for community in model.communities:
#         print(community, community.preferences, community.current_utility, community.platform, community.platform.policies, community.strategy)

# def animation_plot(m, axs):
#     ax1, ax2 = axs
#     ax1.set_title("Virus spread")
#     ax2.set_title(f"Share infected: {m.I}")

#     # Plot stackplot on first axis
#     virus_stackplot(m.output.variables.VirusModel, ax1)
#     color_dict = {0:'b', 1:'r', 2:'g'}
#     colors = [color_dict[c] for c in m.agents.condition]
#     nx.draw_circular(m.network.graph, node_color=colors,
#                      node_size=50)

# fig, axs = plt.subplots(1, 2, figsize=(8, 4)) # Prepare figure
# animation = ap.animate(VirusModel(parameters), fig, axs, animation_plot)

# def virus_stackplot(data, ax):
#     """ Stackplot of people's condition over time. """
#     x = data.index.get_level_values('t')
#     y = [data[var] for var in ['I', 'S', 'R']]

#     sns.set()
#     ax.stackplot(x, y, labels=['Infected', 'Susceptible', 'Recovered'],
#                  colors = ['r', 'b', 'g'])

#     ax.legend()
#     ax.set_xlim(0, max(1, len(x)-1))
#     ax.set_ylim(0, 1)
#     ax.set_xlabel("Time steps")
#     ax.set_ylabel("Percentage of population")

# fig, ax = plt.subplots()
# virus_stackplot(results.variables.VirusModel, ax)

# def animation_plot(m, axs):
#     ax1, ax2 = axs
#     ax1.set_title("Virus spread")
#     ax2.set_title(f"Share infected: {m.I}")

#     # Plot stackplot on first axis
#     virus_stackplot(m.output.variables.VirusModel, ax1)

#     # Plot network on second axis
#     color_dict = {0:'b', 1:'r', 2:'g'}
#     colors = [color_dict[c] for c in m.agents.condition]
#     nx.draw_circular(m.network.graph, node_color=colors,
#                      node_size=50, ax=ax2)

# fig, axs = plt.subplots(1, 2, figsize=(8, 4)) # Prepare figure
# parameters['population'] = 50 # Lower population for better visibility
# animation = ap.animate(VirusModel(parameters), fig, axs, animation_plot)

# IPython.display.HTML(animation.to_jshtml())

