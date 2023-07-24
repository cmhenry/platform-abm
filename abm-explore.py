#!/usr/bin/env python
# coding: utf-8

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
    
    _kind = 'community'

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

    def utility(self,policies):
        """ calculate utility over a given platform from utility fn """
        # basic utility function, no sub-game
        if self.p.p_type == 'binary':
            if len(self.preferences) != len(policies):
                raise ValueError("agent must have complete preferences over vector of platform policies")
            utility = sum(pref == pol for pref,pol in zip(self.preferences, policies))
        elif self.p.p_type == 'non-binary':
            utility = -np.square(self.preferences - policies)
        return(utility)

    def update_utility(self):
        """ updated utility from current platform """
        self.current_utility = self.utility(self.platform.policies)
    
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
    
    _kind = 'platform'
    
    def setup(self):
        """ initialize new variables at platform creation. """
        # set policies
        if self.p.p_type == 'binary':
            self.policies = [random.choice([0, 1]) for _ in range(self.p.p_space)]
        elif self.p.p_type == 'non-binary':
            self.policies = random.randrange(self.p.p_space)

        self.communities = []
        self.community_preferences = []
        # self.ls_utilities = {}
    
    def add_community(self, community):
        """ add community to platform """
        self.communities.append(community)
        
    def rm_community(self, community):
        """ rm community from platform """
        self.communities.remove(community)

    # def community_utilities(self):
    #     """ obtain utilities for attached communities """
    #     for community in self.communities:
    #         self.ls_utilities[community.id] = community.utility(self)
    
    def aggregate_preferences(self):
        """ build an array of community preferences """
        self.community_preferences = np.zeros(shape=(len(self.communities),len(self.policies)))
        
        for idx,community in enumerate(self.communities):
            self.community_preferences[idx] = community.preferences

    def direct_poll(self):
        """ poll individual policies """
        n_policies = len(self.policies)
        
        # ensure aggregate preferences are updated
        self.aggregate_preferences()

        # generate empty list for votes
        votes = [0] * n_policies

        # compile votes
        for col_idx in range(n_policies):
            votes[col_idx] = np.sum(self.community_preferences[:, col_idx] == 0)
        
        return votes
    

    def create_coalitions(self):
        """ initiate coalitions """

        # generate random coalitions
        self.coalitions=np.zeros(shape=(3,10))
        for idx, coalition in enumerate(self.coalitions):
            self.coalitions[idx] = [np.random.choice([0,1]) for _ in range(self.p.p_space)]
    
    def fitness(self, coalition):
        """ check retrieve fitness for each coalition """
        ## fitness: sum of utility gained each from communities
        ## coalitions attempt to maximize community utility
        
        fitness = 0
        
        # iterate over communities & obtain utility
        for community in self.communities:
            utility = community.utility(coalition)
            fitness = fitness + utility
            
        return(fitness)

    def coalition_mutate(self, coalition):
        # iterations paramater = self.p.search_steps
        # perturbations parameter = self.p.perturbations

        # aggregate preferences
        self.aggregate_preferences()

        # assess initial fitness
        fitness = self.fitness(coalition)

        for i in range(self.p.search_steps):
            # create new coalition
            new_coalition = coalition

            # randomly select variables to flip
            variables_to_flip = np.random.choice(new_coalition, 
                                                 self.p.perturbations, 
                                                 replace = False)

            # flip variables inline
            new_coalition = [new_coalition[index] ^ 1 if index in variables_to_flip 
                else value for index, value in enumerate(new_coalition) ]

            # assess fitness
            new_fitness = self.fitness(new_coalition)

            # determine new coalition
            if new_fitness > fitness:
                coalition = new_coalition

        # return new coalition after iterations
        return coalition    
    
    def coalition_poll(self):
        """ gather coalition votes """
        
        communities = p.communities
        coalitions = p.coalitions
        
        votes = []
        
        for community in communities:
            min_utility = float('inf')
            vote_index = None
            
            for idx, coalition in enumerate(coalitions):
                utility = sum(community.preferences == coalition)
                # utility = community.utility(coalition)
                if utility < min_utility:
                    min_utility = utility
                    vote_index = idx
                    
            votes.append(vote_index)
                                      

    def election(self):
        """ election mechanism """

        if(self.p.institution == 'direct'):
            ## gather votes
            votes = self.direct_poll()

            ## set threshold
            threshold = np.floor(0.5 * len(self.communities))
            num_policies = len(self.policies)

            ## count votes & modify policies
            for i in range(num_policies):
                if votes[i] < threshold:
                    ## invert policy if it's below the threshold
                    self.policies[i] = self.policies[i] ^ 1

        if(self.p.institution == 'coalition'):
            ## generate coaltions
            self.create_coalitions()

            ## adapt coaltions
            for coalition in self.coalitions:
                coalition = self.coalition_mutate(coalition)

            ## gather votes
            votes = []
            for i in range(len(self.coalitions)):
                votes.append(self.direct_vote(self.coalitions[i]))

            test = model.platforms[0]
            test.direct_vote()
            test.direct_vote(test.policies)
        

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

