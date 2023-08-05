#!/usr/bin/env python
# coding: utf-8

# Model design
import agentpy as ap
# import networkx as nx
import numpy as np
import pandas as pd
# from surprise import SVD
# from surprise import Dataset
# from surprise import Reader
from sklearn.cluster import KMeans
import random
from collections import Counter

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

class Community(ap.Agent):
    '''
    community characteristics:
        preferences (binary [0,1,...] or continuous range(parameter p_space))
        utility function -(community pref - platform policy)^2
    '''
    
    # _type = 'community'

    def setup(self):
        """ Initialize a new variable at agent creation. """
        self.type = 'mainstream'
        # set preferences
        self.preferences = np.array([random.choice([0, 1]) for _ in range(self.p.p_space)]) # int
        # set initial vars
        self.current_utility = 0
        self.platform = ''
        self.strategy = ''
        self.candidates =[]
        self.group = ''
        self.moves = 0

    def utility(self,policies):
        """ calculate utility over a given platform from utility fn """
        # basic utility function, no sub-game
        if len(self.preferences) != len(policies):
            raise ValueError("agent must have complete preferences over vector of platform policies")
        utility = sum(pref == pol for pref,pol in zip(self.preferences, policies))
        # else:
        #     utility = sum(pref == pol for pref,pol in zip(self.preferences, 
        #         self.platform.group_policies[self.group]))
        return(utility)

    def update_utility(self):
        """ updated utility from current platform """
        # obtain utility from current platform
        if self.platform.institution == "algorithmic":
            c_util = self.utility(self.platform.group_policies[self.group][1])
        else:
            c_util = self.utility(self.platform.policies)

        # extremist vampirism!
        e_util = 0
        if self.type == 'extremist':
            for neighbor in self.platform.communities:
                # if neighbor.type == 'mainstream': e_util += (0.10 * self.p.p_space // 1)
                if neighbor.type == 'mainstream': e_util += 1
        elif self.type == 'mainstream':
            for neighbor in self.platform.communities:
                # if neighbor.type == 'extremist': e_util -= (0.10 * self.p.p_space // 1)
                if neighbor.type == 'extremist': e_util -= 1
        
        self.current_utility = c_util + e_util
    
    def join_platform(self,platform):
        """ join a platform """
        self.moves += 1
        self.platform = platform
    
    def find_new_platform(self):
        """ find candidate platforms """
        self.candidates = []
        for platform in self.model.platforms:
            if platform.institution == 'algorithmic':
                if not platform.group_policies:
                    platform.policies = platform.cold_start_policies()
                    new_policy = random.choice(platform.policies)
                    if self.utility(new_policy) > self.current_utility:
                        self.candidates.append(platform)
                else:
                    for group_policy in platform.group_policies.values():
                        new_policy = group_policy[1]
                        if self.utility(new_policy) > self.current_utility:
                            self.candidates.append(platform)
            else:
                new_policy = platform.policies
                if self.utility(new_policy) > self.current_utility:
                        self.candidates.append(platform)
            
    
    def set_strategy(self):
        """ compare utilities and pick new platform """
        # current_utility = self.update_utility(self.platform)
        for platform in self.model.platforms:
            if platform.institution == 'algorithmic':
                # if platform has no group policies bc it is empty, force it to kickstart
                if not platform.group_policies:
                    platform.policies = platform.cold_start_policies()
                    new_policy = random.choice(platform.policies)
                else:
                    new_policy = platform.group_policies[random.choice(list(platform.group_policies.keys()))][1]
            else:
                new_policy = platform.policies
            
            if self.utility(new_policy) > self.current_utility:
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
    
    _type = 'platform'
    
    def setup(self):
        """ initialize new variables at platform creation. """
        self.institution = ''
        self.policies = []
        self.communities = []
        self.community_preferences = []
        
        # self.ls_utilities = {}
    
    def add_community(self, community):
        """ add community to platform """
        self.communities.append(community)
        
    def rm_community(self, community):
        """ rm community from platform """
        self.communities.remove(community)

    def aggregate_preferences(self):
        """ build an array of community preferences """
        self.community_preferences = np.zeros(shape=(len(self.communities),self.p.p_space), dtype=int)
        
        for idx,community in enumerate(self.communities):
            self.community_preferences[idx] = community.preferences

### DIRECT VOTING INSTITUTION ###

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
    
### MOVEMENT/COALITION INSTITUTION ###

    def create_coalitions(self):
        """ initiate coalitions """

        # generate random coalitions
        self.coalitions=np.zeros(shape=(self.p.coalitions,self.p.p_space), dtype=int)
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
                                                 self.p.mutations, 
                                                 replace = False)\

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
        
        communities = self.communities
        coalitions = self.coalitions
        
        votes = []
        
        for community in communities:
            min_utility = float('inf')
            vote_index = None
            
            for idx, coalition in enumerate(coalitions):
                utility = sum(community.preferences == coalition)
                if utility < min_utility:
                    min_utility = utility
                    vote_index = idx
                    
            votes.append(vote_index)
        
        return(votes)
                                      
### ALGORITHMIC INSTITUTION ###

    def group_communities(self):
        """ sort communities into groups """
        # fallback conditions if platform is empty
        if len(self.communities) == 0:
            return
        # aggregate preferences
        self.aggregate_preferences()

        # check if platform has enough communities
        if len(self.communities) <= self.p.svd_groups:
            svd_groups = len(self.communities)
        else:
            svd_groups = self.p.svd_groups
        # generate kmeans
        kmeans = KMeans(n_clusters=svd_groups, random_state=0,n_init=2)
        # apply kmeans
        groups = kmeans.fit_predict(self.community_preferences)

        self.grouped_communities = [[] for _ in range(svd_groups)]
        for i, group_id in enumerate(groups):
            self.grouped_communities[group_id].append(self.communities[i])
            self.communities[i].group = group_id

    
    def cold_start_policies(self):
        """ construct policy bundles """
        bundles = np.array(np.random.randint(2, size=(5,self.p.p_space))) # int
        return bundles
    
    def rate_policies(self):
        """ serve policy bundldes to community groups """
        self.ui_array = []

        for group_idx, group in enumerate(self.grouped_communities):
            for community in group:
                for bundle_idx, bundle in enumerate(self.policies):
                    fitness = community.utility(bundle)
                    self.ui_array.append([community.id, group_idx, bundle, fitness])
    
    def set_group_policies(self):
        """ set group policies so communities can retrieve utilities """
        highest_ratings = {}

        for community, group, bundle, rating in self.ui_array:
            if group not in highest_ratings:
                highest_ratings[group] = rating,bundle
            else:
                current_rating = highest_ratings[group][0]
                if rating > current_rating:
                    highest_ratings[group] = rating,bundle
        
        self.group_policies = highest_ratings
                    

    # def svd_group(self):
    #     """ svd for each group """
        
    #     # conver to dataframe to make pivot easier
    #     self.ui_df = pd.DataFrame(self.ui_array, columns = ['community','group_id','bundle_id','fitness'])
        
    #     # build ui_mats for each group
    #     pivoted_ui_mat = {}
    #     groups = self.ui_df['group_id'].unique()
        
    #     # svd for ui_mats
    #     self.group_svd = {}
    #     for group in groups:
    #         group_df = self.ui_df[self.ui_df['group_id'] == group]
    #         group_mat = group_df.pivot(index='community',columns='bundle_id',values='fitness')
    #         pivoted_ui_mat[group] = group_mat
    #         self.group_svd[group] = np.linalg.svd(pivoted_ui_mat[group], full_matrices=False)
        
    #     reader = Reader(rating_scale=(0,50))
    #     dataset = Dataset.load_from_df(ptest.ui_df[['community','bundle_id','fitness']], reader)
    #     trainset = dataset.build_full_trainset()
    #     algo = SVD()
    #     algo.fit(trainset)
    #     algo.predict(uid = str(679), iid = str(0), r_ui=34.0, verbose = True)

    #     dataset.raw_ratings
        
    # def recommmender(self):
    #     """ use svd to recommend new bundle """

    def election(self):
        """ election mechanism """

        if(self.institution == 'direct'):
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

        if(self.institution == 'coalition'):
            ## generate coalitions
            self.create_coalitions()
            
            ## adapt coaltions
            for coalition in self.coalitions:
                coalition = self.coalition_mutate(coalition)

            ## gather votes
            votes = self.coalition_poll()
            
            ## count votes and reset policy
            count = Counter(votes)
            winners = [item for item, freq in count.items() if freq == max(count.values())]
            new_policies = self.coalitions[random.choice(winners)]
            self.policies = new_policies

        if(self.institution == 'algorithmic'):
            ## produce new content slate
            self.policies = self.cold_start_policies()

            ## predict new policy ratings by SVD
            self.group_communities()
            self.rate_policies()

            ## set group slates
            self.set_group_policies()
        


class MiniTiebout(ap.Model):

### SETUP ###

    def setup(self):
        """ Initialize the agents and network of the model. """
        # prepare network
        # graph = nx.Graph()
        
        # model = MiniTiebout(parameters)
        # model.setup()
        # model.platforms.test = ap.AttrIter(['1','2'])
        # model.platforms.random(n=2)
        
        # add communities to network
        self.communities = ap.AgentList(self, self.p.n_comms, Community)
        # for community in self.communities:
        #     graph.add_node(community)
        # mix communities if necessary
        if self.p.extremists == "yes":
            extremists, mainstream = self.setup_mix_agents_by_percentage(self.communities,self.p.percent_extremists)
            self.setup_community_types(extremists)
            
        
        # add platforms to network
        self.platforms = ap.AgentList(self, self.p.n_plats, Platform)
        # mix platforms if necessary
        if self.p.institution == 'mixed':
            sub_platforms = self.setup_mix_agents_by_split(self.platforms,3)
            self.setup_platform_institutions(sub_platforms)
        else: self.platforms.institution = self.p.institution  
        self.setup_platform_policies()
        # mix platforms for extremism if necessary
        if self.p.extremists == 'yes':
            extremists, mainstream = self.setup_mix_agents_by_percentage(self.platforms, self.p.percent_extremists)
            self.setup_platform_types(extremists)
            
            
        # for platform in self.platforms:
        #     graph.add_node(platform)
        
        # randomly add communities to platforms
        for community in self.communities:
            platform = self.random.choice(self.platforms)
            community.join_platform(platform)
            platform.add_community(community)
            # graph.add_edge(community, platform)

        # setup algorithmic platforms
        for platform in self.platforms:
            if platform.institution == 'algorithmic':
                # group communities based on knn
                platform.grouped_communities = []
                platform.group_communities()
                # generate community-content ratings
                platform.rate_policies()
                ## set group slates
                platform.set_group_policies()
                # # do initial SVD
                # self.svd_group()
        
        # combine into agentlist & register network
        self.agents = self.communities + self.platforms
        # self.network = self.agents.network = ap.Network(self, graph)
        # self.network.add_agents(self.agents, self.network.nodes)
        
    def setup_mix_agents_by_split(self, agentlist, split):
        # generate primitives to mix list
        remaining = agentlist.copy()
        total = len(remaining)
        sublist_size = total // split
        remainder = total % split
        sublists = []
        
        # begin mixing
        for _ in range(split):
            sublist = []
            for _ in range(sublist_size):
                if remaining:
                    item = random.choice(remaining)
                    remaining.remove(item)
                    sublist.append(item.id)
            sublists.append(sublist)
        
        # ensure remainder are sorted
        for _ in range(remainder):
            if remaining:
                item = random.choice(remaining)
                remaining.remove(item)
                sublists[_].append(item.id)
        
        return(sublists)
    
    def setup_mix_agents_by_percentage(self, agentlist, percentage):
        # sanity check
        if percentage <= 0 or percentage > 100:
            raise ValueError("percentage must be between 0 and 100")
        # generate primitives to mix list
        num_items_to_select = int(len(agentlist) * (percentage / 100))
        selected_items = random.sample(agentlist.id, num_items_to_select)
        unselected_items = [item for item in agentlist.id if item not in selected_items]
        
        return(selected_items, unselected_items)
        
    
    def setup_platform_institutions(self, sub_platforms):
        """ assign platform institutions """
        instlist = ['algorithmic','direct','coalition']
        for sublist_idx, sublist in enumerate(sub_platforms):
            for item in sublist:
                self.platforms.select(self.platforms.id == item).institution = instlist[sublist_idx]

    def setup_platform_types(self, sub_platforms):
        """ assign platform types """
        for id in sub_platforms:
            if self.platforms.select(self.platforms.id == id).institution != 'algorithmic':
                self.platforms.select(self.platforms.id == id).policies = [0 for _ in range(self.p.p_space)]
    
    def setup_platform_policies(self):
        """ set platform policies """
        for platform in self.platforms:
            # set policies
            if platform.institution != 'algorithmic':
            # generate random single policy slate
                platform.policies = np.array([random.choice([0, 1]) for _ in range(self.p.p_space)]) # int
            else:
            # generate random policy slates
                platform.policies = platform.cold_start_policies() 
    
    def setup_community_types(self, extremists):
        """ assign community types """      
        for id in extremists:
            self.communities.select(self.communities.id == id).type = "extremist"
            self.communities.select(self.communities.id == id).preferences = [0 for _ in range(self.p.p_space)]
            

### UPDATE ###

    def update(self):
        """ Record variables after setup and each step. """

        # record community strategies, utilities, and platforms
        # for platform in self.platforms:
        #     # average utility
        #     platform.community_utilities()
        #     avg_utility = sum(platform.ls_utilities.values()) / float(len(platform.ls_utilities))
        #     self.record(f'{"avg_util"}{platform.id}', 
        #         avg_utility)

        # for community in self.communities:
        #     # utility
        #     self.record()

        #     # location
        #     self.record(f'{"loc_com"}{community.id}', 
        #         community.platform)
        #     # strategy
        #     self.record(f'{"strat_com"}{community.id}',
        #         community.strategy)
        #     # utility
        #     self.record(f'{"util_com"}{community.id}',
        #         community.current_utility)
        
        ### OUTCOMES OF INTEREST ###
        # 1. history of platform policies
        # 2. history of community utilities
        # 3. history of community locations

        # platform policies

        
        # check if every community is happy
        if self.p.stop_condition == 'satisficed':
            if self.update_satisficed():
                self.end()
    
    def update_satisficed(self):
        """ check state of communities for equlibrium """
        for community in self.communities:
            if community.strategy == 'move':
                return False
        return True

### STEP ###

    def step(self):
        """ Define the models' events per simulation step. """
        # relocation: update utility
        self.step_update_utility()
        # relocation: search for new platforms
        self.step_relocation()
        # elections: platforms aggregate institutions
        self.step_elections()

    def step_update_utility(self):
        """ function to update all community agent utilities """
        for community in self.communities:
            community.update_utility()
            community.set_strategy()
    
    def step_relocation(self):
        """ function to relocate communities """
        for community in self.communities:
            if community.strategy == 'move':
                # generate list of candidates
                community.find_new_platform()
                # community joins new platform
                new_platform = random.choice(community.candidates)
                community.join_platform(new_platform)
                # add community to platform
                new_platform.add_community(community)
        for platform in self.platforms:
            if platform.institution == 'algorithmic': platform.group_communities()

    def step_elections(self):
        """ function to hold elections """
        for platform in self.platforms:
                platform.election()

### END ###

    def end(self):
        # reporters
        # stability
        self.report('average_moves', 
                    sum(self.communities.moves) / self.p.n_comms)
        # per capita utility
        self.report('average_utility', 
                    sum(self.communities.current_utility) / self.p.n_comms)
        # dealing with extremists
        if self.p.extremists == 'yes':
            # per capita utility extremists
            extremists = self.communities.select(self.communities.type == 'extremist')
            self.report('average_extremist_utility',
                        sum(extremists.current_utility) / len(extremists))
            # per capita utility mainstream
            mainstream = self.communities.select(self.communities.type == 'mainstream')
            self.report('average_mainstream_utility',
                        sum(mainstream.current_utility) / len(mainstream))
        
    # def end_utility_per_platform_type(self):
        
# parameters = {
#     'n_comms': 1000,
#     'n_plats': 10,
#     'p_space': 10,
#     'p_type': 'binary',
#     'steps':50,
#     'institution': 'mixed',
#     'extremists': 'yes',
#     'percent_extremists': 5,
#     'coalitions': 2,
#     'mutations': 2,
#     'search_steps': 10,
#     'svd_groups': 3,
#     'stop_condition': 'steps'
# }

# model = MiniTiebout(parameters)
# results = model.run()


exp_parameters = {
    'n_comms': ap.IntRange(100,1000),
    'n_plats': ap.IntRange(10, 100),
    'p_space': ap.IntRange(10, 100),
    'p_type': 'binary',
    'steps': 50,
    'institution': ap.Values('mixed','algorithmic','direct','coalition'),
    'extremists': ap.Values('yes','no'),
    'percent_extremists': ap.Values(5,10,20,30),
    'coalitions': ap.IntRange(2,10),
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps'
}

sample = ap.Sample(
    exp_parameters,
    n=100,
    method='saltelli',
    calc_second_order=False
)

exp = ap.Experiment(MiniTiebout, sample, iterations=10, record=True)
results = exp.run(n_jobs = -1, verbose=10)
