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

class Community(ap.Agent):
    
    kind = 'community'

    def setup(self):
        """ Initialize a new variable at agent creation. """
        # set preferences
        self.preferences = [random.choice([0, 1]) for _ in range(self.p.p_space)]
        # set initial vars
        self.current_utility = 0
        self.platform = ''
        self.strategy = ''

    def utility(self,platform):
        """ calculate utility over a given platform from utility fn """
        # basic utility function, no sub-game
        if len(self.preferences) != len(platform.policies):
            raise ValueError("agent must have complete preferences over vector of platform policies")

        utility = sum(pref == pol for pref,pol in zip(self.preferences, platform.policies))
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
    
    kind = 'platform'
    
    def setup(self):
        """ initialize new variables at platform creation. """
        # set preferences
        self.policies = [random.choice([0, 1]) for _ in range(self.p.p_space)]
        self.communities = []
        self.community_utilites = {}
    
    def add_community(self, community):
        """ add community to platform """
        self.communities.append(community)
        
    def rm_community(self, community):
        """ rm community from platform """
        self.communities.remove(community)

    def community_utilities(self):
        """ obtain utilities for attached communities """
        for community in self.communities:
            self.community_utilities[community.id] = community.utility(self)
        

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
            if community.strategy != 'move':
                return False
        return True

    def update(self):
        """ Record variables after setup and each step. """
        # update utilities for current platform-community relationships
        for community in self.communities:
            community.update_utility()
            community.set_strategy()
        if self.satisfied():
            self.stop()

#         # Record share of agents with each condition
#         for i, c in enumerate(('S', 'I', 'R')):
#             n_agents = len(self.agents.select(self.agents.condition == i))
#             self[c] = n_agents / self.p.population
#             self.record(c)

#         # Stop simulation if disease is gone
#         if self.I == 0:
#             self.stop()
    
    # def get_agent(self, agent_node):
    #     agent = list(agent_node)[0]
    #     return agent
                
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


    def end(self):
        """ Record evaluation measures at the end of the simulation. """
        pass

#         # Record final evaluation measures
#         self.report('Total share infected', self.I + self.R)
#         self.report('Peak share infected', max(self.log['I']))

parameters = {
    'n_comms': 10,
    'n_plats': 2,
    'p_space': 5,
    'steps':20
}

model = MiniTiebout(parameters)
model.setup()
model.update()
report(model)
model.step()
report(model)
model.update()
report(model)

results = model.run()
def report(model):
    for community in model.communities:
        print(community, community.preferences, community.current_utility, community.platform, community.platform.policies, community.strategy)

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

