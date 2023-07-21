import numpy as np

def count_zeros_in_columns(array):
    num_columns = array.shape[1]

    counts = [0] * num_columns

    for col_idx in range(num_columns):
        counts[col_idx] = np.sum(array[:, col_idx] == 0)

    return counts

# Example usage
array = np.random.randint(2, size=(10, 10))

zero_counts = count_zeros_in_columns(array)
print("Zero counts in each column:", zero_counts)

def generate_random_platform():
    return np.random.randint(2, size=10)

def calculate_utility(community_preferences, platform_policies):
    return np.sum(community_preferences == platform_policies, axis=1)

def mutate(platform, mutation_rate):
    mutated_platform = np.copy(platform)
    for i in range(len(platform)):
        if np.random.rand() < mutation_rate:
            mutated_platform[i] = 1 - mutated_platform[i]
    return mutated_platform

def genetic_algorithm(agent_preferences, num_iterations=10, population_size=100, mutation_rate=0.1):
    num_agents = agent_preferences.shape[0]
    num_variables = agent_preferences.shape[1]

    coalition1_population = [generate_random_platform() for _ in range(population_size)]
    coalition2_population = [generate_random_platform() for _ in range(population_size)]

    for iteration in range(num_iterations):
        coalition1_fitness = np.zeros(population_size)
        coalition2_fitness = np.zeros(population_size)

        for i in range(population_size):
            coalition1_fitness[i] = calculate_utility(agent_preferences, coalition1_population[i]).sum()
            coalition2_fitness[i] = calculate_utility(agent_preferences, coalition2_population[i]).sum()

        # Sort populations based on fitness
        coalition1_population = [x for _, x in sorted(zip(coalition1_fitness, coalition1_population), reverse=True)]
        coalition2_population = [x for _, x in sorted(zip(coalition2_fitness, coalition2_population), reverse=True)]

        # Perform mutation on top-performing platforms
        for i in range(population_size // 2):
            parent1_idx = np.random.randint(population_size // 2)
            parent2_idx = np.random.randint(population_size // 2)

            child1 = mutate(coalition1_population[parent1_idx], mutation_rate)
            child2 = mutate(coalition1_population[parent2_idx], mutation_rate)
            coalition1_population[population_size // 2 + i] = child1
            coalition1_population[population_size // 2 + i + 1] = child2

            child1 = mutate(coalition2_population[parent1_idx], mutation_rate)
            child2 = mutate(coalition2_population[parent2_idx], mutation_rate)
            coalition2_population[population_size // 2 + i] = child1
            coalition2_population[population_size // 2 + i + 1] = child2

    return coalition1_population[0], coalition2_population[0]

# Example usage
num_agents = 100
num_variables = 10

# Generate random agent preferences
agent_preferences = np.random.randint(2, size=(num_agents, num_variables))

# Perform genetic algorithm with mutation
coalition1_platform, coalition2_platform = genetic_algorithm(agent_preferences)

print("Coalition 1 platform:", coalition1_platform)
print("Coalition 2 platform:", coalition2_platform)
