import minitiebout
import networkx as nx 
import matplotlib.pyplot as plt

### Experiment 3 trial 1: multiple platform institutional comparisons + extremists

param_3a_t1_direct = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'direct',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 5,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_3b_t1_coalition = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'coalition',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 2,
    'stop_condition': 'steps',
    'seed': 1999
}

param_3c_t1_algorithm = {
    'n_comms': 100,
    'n_plats': 5,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'algorithm',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3a_t1 = minitiebout.MiniTiebout(param_3a_t1_direct)
model_3b_t1 = minitiebout.MiniTiebout(param_3b_t1_coalition)
model_3c_t1 = minitiebout.MiniTiebout(param_3c_t1_algorithm)

results_3a_t1 = model_3a_t1.run()
results_3b_t1 = model_3b_t1.run()
results_3c_t1 = model_3c_t1.run()

results_3a_t1.reporters 
results_3b_t1.reporters 
results_3c_t1.reporters 

extremists = model_3a_t1.communities.select(model_3a_t1.communities.type == 'extremist')
# self.report('average_extremist_utility',
#                         sum(extremists.current_utility) / len(extremists))
# per capita utility mainstream
mainstream = model_3a_t1.communities.select(model_3a_t1.communities.type == 'mainstream')
# self.report('average_mainstream_utility',
#                         sum(mainstream.current_utility) / len(mainstream))



### with 5 percent extremists
# >>> results_3a_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          37.44             5.88                       26.4                         4.8
# >>> results_3b_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          44.77             4.71                       23.8                    3.705263
# >>> results_3c_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          29.84             6.56                       17.8                    5.968421

### with 10 percent extremists
# >>> results_3a_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          40.69             5.65                       23.0                    3.722222
# >>> results_3b_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          44.13             4.91                       20.5                    3.177778
# >>> results_3c_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999          34.72             6.43                       20.4                    4.877778

### with 15 percent extremists
# >>> results_3a_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999           43.5             5.65                  24.666667                    2.294118
# >>> results_3b_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999           43.5             4.32                       20.6                    1.447059
# >>> results_3c_t1.reporters 
#    seed  average_moves  average_utility  average_extremist_utility  average_mainstream_utility
# 0  1999           43.5             5.42                  18.333333                    3.141176

### Experiment 3b: mixed institutions, multiple platforms + extremists

param_3b_mixed = {
    'n_comms': 900,
    'n_plats': 3,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 10,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b = minitiebout.MiniTiebout(param_3b_mixed)

results_3b = model_3b.run()

model_3b.reporters

directcomms = model_3b.communities.select(model_3b.communities.platform.institution == 'direct')
sum(directcomms.select(directcomms.type == 'extremist').current_utility) / len(directcomms.select(directcomms.type == 'extremist'))
sum(directcomms.select(directcomms.type == 'mainstream').current_utility) / len(directcomms.select(directcomms.type == 'mainstream'))

### with 5 percent extremists + 3 platforms
# >>> model_3b.reporters
# {'seed': 1999, 'average_moves': 48.5, 'average_utility': 5.417777777777777, 'n_direct_comms': 195, 
# 'n_coalition_comms': 180, 'n_algo_comms': 525, 'ratio_direct': 0.21666666666666667, 'ratio_coalition': 0.2,
#  'ratio_algo': 0.5833333333333334, 'util_direct': 1237, 'util_coalition': 647, 'util_algo': 2992, 
#  'avg_utility_direct': 6.343589743589743, 'avg_utility_coalition': 3.5944444444444446, 
#  'avg_utility_algo': 5.699047619047619, 'average_extremist_utility': 297.4, 
#  'average_mainstream_utility': -9.949707602339181}
# avg direct ext 179 avg direct main -10.14
# avg coal ext 191 avg coal main -9.79
# avg algo ext 503 avg algo main -9.93

### with 10 percent extremists + 3 platforms
# >>> model_3b.reporters
# {'seed': 1999, 'average_moves': 46.0, 'average_utility': 5.513333333333334, 'n_direct_comms': 193, 
# 'n_coalition_comms': 197, 'n_algo_comms': 510, 'ratio_direct': 0.21444444444444444, 
# 'ratio_coalition': 0.21888888888888888, 'ratio_algo': 0.5666666666666667, 'util_direct': 232, 
# 'util_coalition': 2406, 'util_algo': 2324, 'avg_utility_direct': 1.2020725388601037, 
# 'avg_utility_coalition': 12.213197969543147, 'avg_utility_algo': 4.556862745098039, 
# 'average_extremist_utility': 269.6, 'average_mainstream_utility': -23.82962962962963}
# avg direct ext 162 avg direct main -23.83
# avg coal ext 172 avg coal main -23.52
# avg algo ext 496 avg algo main -23.93

### with 15 percent extremists + 3 platforms
# >>> model_3b.reporters
# >>> model_3b.reporters
# {'seed': 1999, 'average_moves': 43.5, 'average_utility': 5.174444444444444, 'n_direct_comms': 212, 
# 'n_coalition_comms': 186, 'n_algo_comms': 502, 'ratio_direct': 0.23555555555555555, 
# 'ratio_coalition': 0.20666666666666667, 'ratio_algo': 0.5577777777777778, 'util_direct': 4200, 
# 'util_coalition': -803, 'util_algo': 1260, 'avg_utility_direct': 19.81132075471698, 
# 'avg_utility_coalition': -4.317204301075269, 'avg_utility_algo': 2.50996015936255, 
# 'average_extremist_utility': 257.68148148148146, 'average_mainstream_utility': -39.38562091503268}
# avg direct ext 172 avg direct main -36.84
# avg coal ext 169 avg coal main -38.98
# avg algo ext 437 avg algo main -39.23

param_3b_t2_mixed = {
    'n_comms': 300,
    'n_plats': 9,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 15,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b_t2 = minitiebout.MiniTiebout(param_3b_t2_mixed)

results_3b_t2 = model_3b_t2.run()

model_3b_t2.reporters

directcomms = model_3b_t2.communities.select(model_3b_t2.communities.platform.institution == 'algorithmic')
sum(directcomms.select(directcomms.type == 'extremist').current_utility) / len(directcomms.select(directcomms.type == 'extremist'))
sum(directcomms.select(directcomms.type == 'mainstream').current_utility) / len(directcomms.select(directcomms.type == 'mainstream'))

### with 5 percent extremists + 9 platforms
# >>> model_3b_t2.reporters
# {'seed': 1999, 'average_moves': 43.61, 'average_utility': 5.69, 'n_direct_comms': 60, 'n_coalition_comms': 55,
#  'n_algo_comms': 185, 'ratio_direct': 0.2, 'ratio_coalition': 0.18333333333333332, 'ratio_algo': 0.6166666666666667,
#   'util_direct': 417, 'util_coalition': 349, 'util_algo': 941, 'avg_utility_direct': 6.95, 
#   'avg_utility_coalition': 6.345454545454546, 'avg_utility_algo': 5.0864864864864865, 
#   'average_extremist_utility': 34.4, 'average_mainstream_utility': 4.178947368421053}
#  'average_mainstream_utility': 4.196491228070175}
# avg direct ext 29.67 avg direct main 4.42
# avg coal ext 30 avg coal main 3.98
# avg algo ext 47 avg algo main 4.16

### with 10 percent extremists + 9 platforms
# >>> model_3b_t2.reporters
# {'seed': 1999, 'average_moves': 45.99333333333333, 'average_utility': 5.483333333333333, 
# 'n_direct_comms': 67, 'n_coalition_comms': 63, 'n_algo_comms': 170, 'ratio_direct': 0.22333333333333333, 
# 'ratio_coalition': 0.21, 'ratio_algo': 0.5666666666666667, 'util_direct': 280, 'util_coalition': 235,
#  'util_algo': 1130, 'avg_utility_direct': 4.17910447761194, 'avg_utility_coalition': 3.7301587301587302,
#   'avg_utility_algo': 6.647058823529412, 'average_extremist_utility': 41.9, 
#   'average_mainstream_utility': 1.4370370370370371}
# avg direct ext 26.50 avg direct main 2.07
# avg coal ext 21.22 avg coal main 1.94
# avg algo ext 60.00 avg algo main 1.62

### with 15 percent extremists + 9 platforms
# >>> model_3b_t2.reporters
# {'seed': 1999, 'average_moves': 43.233333333333334, 'average_utility': 5.223333333333334, 
# 'n_direct_comms': 80, 'n_coalition_comms': 53, 'n_algo_comms': 167, 'ratio_direct': 0.26666666666666666, 
# 'ratio_coalition': 0.17666666666666667, 'ratio_algo': 0.5566666666666666, 'util_direct': 403, 
# 'util_coalition': 282, 'util_algo': 882, 'avg_utility_direct': 5.0375, 'avg_utility_coalition': 5.320754716981132,
#  'avg_utility_algo': 5.281437125748503, 'average_extremist_utility': 33.022222222222226, 
#  'average_mainstream_utility': 0.3176470588235294}
# avg direct ext 24.76 avg direct main -0.28
# avg coal ext 18.53 avg coal main 1.025
# avg algo ext 54.93 avg algo main 0.38

param_3b_t3_mixed = {
    'n_comms': 900,
    'n_plats': 15,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 10,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b_t3 = minitiebout.MiniTiebout(param_3b_t3_mixed)

results_3b_t3 = model_3b_t3.run()

model_3b_t3.reporters

directcomms = model_3b_t3.communities.select(model_3b_t3.communities.platform.institution == 'direct')
sum(directcomms.select(directcomms.type == 'extremist').current_utility) / len(directcomms.select(directcomms.type == 'extremist'))
sum(directcomms.select(directcomms.type == 'mainstream').current_utility) / len(directcomms.select(directcomms.type == 'mainstream'))

directplats = model_3b_t3.platforms.select(model_3b_t3.platforms.institution == 'direct')
algoplats = model_3b_t3.platforms.select(model_3b_t3.platforms.institution == 'algorithmic')
coalplats = model_3b_t3.platforms.select(model_3b_t3.platforms.institution == 'coalition')

def draw_graph(platformlist, index=0):
    """ draw networkgraphs of platform + communities """
    platform = platformlist[index]
    G = nx.Graph()
    G.add_node(platform)
    for community in platform.communities:
        G.add_nodes_from([(community, {'type':community.type})])
        G.add_edge(platform, community)

    color_map=[]
    for node in G:
        if node.type == 'extremist':
            color_map.append('red')
        elif node.type == 'mainstream':
            color_map.append('green')
        else:
            color_map.append('blue')

    plt.clf()
    nx.draw(G, with_labels=False, node_color=color_map)
    plt.savefig("platform%s.png" % platform.id)
    
for idx in range(len(algoplats)):
    draw_graph(algoplats,idx)

### mixed + 15 plats + 5 percent extremists
# >>> model_3b_t3.reporters
# {'seed': 1999, 'average_moves': 48.062222222222225, 'average_utility': 5.4655555555555555, 'n_direct_comms': 185, 
# 'n_coalition_comms': 197, 'n_algo_comms': 518, 'ratio_direct': 0.20555555555555555, 
# 'ratio_coalition': 0.21888888888888888, 'ratio_algo': 0.5755555555555556, 'util_direct': 959, 
# 'util_coalition': 1082, 'util_algo': 2878, 'avg_utility_direct': 5.183783783783784, 
# 'avg_utility_coalition': 5.49238578680203, 'avg_utility_algo': 5.555984555984556, 
# 'average_extremist_utility': 64.82222222222222, 'average_mainstream_utility': 2.3415204678362573}
# avg direct ext 47.2 avg direct main 2.78
# avg coal ext 36.55 avg coal main 2.36
# avg algo ext 105.12 avg algo main 2.177

### mixed + 15 plats + 10 percent extremists
# >>> model_3b_t3.reporters
# {'seed': 1999, 'average_moves': 46.0, 'average_utility': 5.174444444444444, 'n_direct_comms': 191, 
# 'n_coalition_comms': 198, 'n_algo_comms': 511, 'ratio_direct': 0.21222222222222223, 'ratio_coalition': 0.22, 
# 'ratio_algo': 0.5677777777777778, 'util_direct': 1200, 'util_coalition': 945, 'util_algo': 2512, 
# 'avg_utility_direct': 6.282722513089006, 'avg_utility_coalition': 4.7727272727272725, 
# 'avg_utility_algo': 4.915851272015655, 'average_extremist_utility': 57.01111111111111, 
# 'average_mainstream_utility': -0.5851851851851851}
# avg direct ext 40 avg direct main -0.50
# avg coal ext 33.53 avg coal main -0.36
# avg algo ext 101.61 avg algo main -0.69

### mixed + 15 plats + 15 percent extremists
# >>> model_3b_t3.reporters
# {'seed': 1999, 'average_moves': 43.5, 'average_utility': 5.363333333333333, 'n_direct_comms': 213, 
# 'n_coalition_comms': 178, 'n_algo_comms': 509, 'ratio_direct': 0.23666666666666666, 
# 'ratio_coalition': 0.19777777777777777, 'ratio_algo': 0.5655555555555556, 'util_direct': 880, 
# 'util_coalition': 521, 'util_algo': 3426, 'avg_utility_direct': 4.131455399061033, 
# 'avg_utility_coalition': 2.9269662921348316, 'avg_utility_algo': 6.730844793713163, 
# 'average_extremist_utility': 62.28148148148148, 'average_mainstream_utility': -4.681045751633987}
# avg direct ext 32.77 avg direct main -4.42
# avg coal ext 35.06 avg coal main -4.38
# avg algo ext 106.51 avg algo main -4.86

param_3b_jumbo_mixed = {
    'n_comms': 1800,
    'n_plats': 30,
    'p_space': 10,
    'p_type': 'binary',
    'steps':50,
    'institution': 'mixed',
    'extremists': 'yes',
    'percent_extremists': 5,
    'coalitions': 3,
    'mutations': 2,
    'search_steps': 10,
    'svd_groups': 3,
    'stop_condition': 'steps',
    'seed': 1999
}

model_3b_jumbo = minitiebout.MiniTiebout(param_3b_jumbo_mixed)

results_3b_jumbo = model_3b_jumbo.run()

results_3b_jumbo.reporters

### with 5 percent extremists + 6 platforms + 6 coalitions + 1500 communities
# >>> results_3b_jumbo.reporters
#    seed  average_moves  average_utility  n_direct_comms  n_coalition_comms  n_algo_comms  ratio_direct  ratio_coalition  ratio_algo  \
# 0  1999           48.5         5.475333             307                290           903      0.204667         0.193333       0.602   

#    avg_utility_direct  avg_utility_coalition  avg_utility_algo  average_extremist_utility  average_mainstream_utility  
# 0           15.990228               3.106897          5.436323                 243.626667                   -7.058947 
## QUICKTAKE: direct voting on policies -> more utility for communities, both types; coalition utility goes up when
## there are more coalition slots available and communities can shrink; extremists netting higher overall utility
## than mainstream; although some extremists end up on "extremist" platforms with favorable policies, big utility
## gains for extremists are those who stick around non-extremist platforms with just-unfavorable policies &
## steal utility from mainstream