import numpy as np


# Generate random user-term matrix data as binary preferences
num_users = 1000
num_terms = 10
user_term_matrix = np.random.randint(2, size=(num_users, num_terms))
print("User-Term Matrix:")
print(user_term_matrix)

# Perform Singular Value Decomposition (SVD) on the user-term matrix
U, S, Vt = np.linalg.svd(user_term_matrix, full_matrices=False)

# Reconstruct the original matrix from the SVD components
# S_diag = np.zeros((num_users, num_terms))
# S_diag[:num_users, :num_users] = np.diag(S)
# reconstructed_matrix = np.dot(U, np.dot(S_diag, Vt))
# print("\nReconstructed Matrix:")
# print(np.round(reconstructed_matrix).astype(int))

# Us = np.dot(U, np.diag(S))
# skV = np.dot(np.diag(S),Vt)
# UsV = np.dot(Us, skV)

# Select three users from the SVD
selected_users_indices = [0, 2, 4]
selected_users_U = U[selected_users_indices, :]
selected_users_preferences = user_term_matrix[selected_users_indices, :]

# Calculate the group-specific latent factors
group_specific_latent_factors = np.dot(selected_users_preferences.T, selected_users_U)

# Modify U, S, and Vt matrices to obtain group-specific latent factors
group_U = np.copy(U)
group_U[selected_users_indices, :] = group_specific_latent_factors.T

# The group-specific S matrix can be obtained by simply taking the diagonal matrix of the group-specific singular values
group_S = np.diag(S)

# The group-specific Vt matrix remains the same since it represents the term-feature matrix, which is independent of user preferences

# Reconstruct the user-term matrix for the group using the modified U, S, and Vt matrices
group_reconstructed_matrix = np.dot(selected_users_U, np.dot(group_S, Vt))

print("\nGroup-Specific Reconstructed Matrix:")
print(np.round(group_reconstructed_matrix).astype(int))

def recommend_items_to_new_user(new_user_preferences, selected_users_preferences, U, Vt, num_recommendations=5):
    # Calculate the group-specific latent factors
    # group_specific_latent_factors = np.linalg.pinv(selected_users_preferences).dot(U)

    # Calculate the new user's latent factors
    new_user_latent_factors = new_user_preferences.dot(group_specific_latent_factors)

    # Obtain the predicted user-item interaction
    predicted_interaction = new_user_latent_factors.dot(Vt)

    # Rank the items based on predicted interaction and recommend top N items
    recommended_item_indices = np.argsort(predicted_interaction)[::-1][:num_recommendations]

    return recommended_item_indices

# Example usage
# Assume you have U, Vt, and selected_users_preferences from previous SVD calculations
num_items = 100

# Simulate a new user's preferences (binary array of length num_items)
new_user_preferences = np.random.randint(2, size=num_items)

# Recommend top 3 items to the new user
recommended_items = recommend_items_to_new_user(new_user_preferences, selected_users_preferences, U, Vt, num_recommendations=3)

print("Recommended Item Indices:", recommended_item_indices)


import numpy as np

class Agent:
    def __init__(self, preferences):
        self.preferences = preferences
        self.utility = 0.0

    def calculate_utility(self, policy_list):
        self.utility = np.sum(self.preferences != policy_list)

class Node:
    def __init__(self, agent_list, num_policies=10):
        self.agent_list = agent_list
        self.policies = []
        self.num_policies = num_policies

    def generate_random_policies(self):
        self.policies = [np.random.randint(2, size=10) for _ in range(self.num_policies)]

    def group_agents(self):
        groups = [[] for _ in range(self.num_policies)]
        for agent in self.agent_list:
            group_index = np.random.randint(self.num_policies)
            groups[group_index].append(agent)
        return groups

    def perform_svd(self, groups):
        # Combine preferences of agents in each group into a matrix
        group_preferences = [np.array([agent.preferences for agent in group]) for group in groups]
        combined_preferences = [np.vstack(group) for group in group_preferences]

        # Perform SVD for each group separately
        latent_factors_list = []
        self.Vt = []
        for preference_matrix in combined_preferences:
            U, S, Vt = np.linalg.svd(preference_matrix, full_matrices=False)
            # Store the latent factors for this group
            latent_factors_list.append(U)
            self.Vt.append(Vt)

        # Combine the latent factors of all groups into a single matrix
        latent_factors = np.vstack(latent_factors_list)

        return latent_factors

    def predict_best_policies(self, latent_factors):
        predicted_policies = []
        for idx, group in enumerate(groups):
            predicted_policies.append(np.dot(latent_factors[idx], node.Vt[idx]))
        return predicted_policies

    def new_agents_join(self, num_new_agents=5):
        new_agents = [Agent(np.random.randint(2, size=10)) for _ in range(num_new_agents)]
        self.agent_list.extend(new_agents)

    def sort_new_agents_into_groups(self, new_agents, latent_factors):
        groups = [[] for _ in range(self.groups)]
        for agent in new_agents:
            agent = new_agents[0]
            group_index = np.argmax(np.dot(agent.preferences, latent_factors.T))
            groups[group_index].append(agent)
        return groups

    def calculate_group_utility(self, groups, best_policies):
        for i, group in enumerate(groups):
            for agent in group:
                agent.calculate_utility(best_policies[i])

# Function to perform the cycle steps
def perform_cycle(node):
    # Step 1
    node.generate_random_policies()

    # Step 2
    groups = node.group_agents()

    # Step 3 - Calculate utility for agents in each group
    for idx,group in enumerate(groups):
        for agent in group:
            agent.calculate_utility(node.policies[idx])

    # Step 4 - Perform SVD
    latent_factors = node.perform_svd(groups)

    # Step 5 - Predict best policies for each group
    best_policies = node.predict_best_policies(latent_factors)

    # Step 6 - New agents join the Node
    node.new_agents_join()

    # Step 7 - Sort new agents into groups using SVD latent factors
    new_agents = node.agent_list[-5:]  # Assuming last 5 agents are the new ones
    new_groups = node.sort_new_agents_into_groups(new_agents, latent_factors)

    # Step 8 - Calculate utility for new agents in each group
    node.calculate_group_utility(new_groups, best_policies)

# Example usage
agent_list = [Agent(np.random.randint(2, size=10)) for _ in range(1000)]
node = Node(agent_list)
perform_cycle(node)


similar_lists = []
initial_list = np.array([random.choice([0, 1]) for _ in range(10)])
distance_range = (1,10)

for _ in range(10):
    new_list = initial_list.copy()
    hamming_distance = random.randint(*distance_range)