import random
import numpy as np
import matplotlib.pyplot as plt
import time
from SetCoveringProblemCreator import SetCoveringProblemCreator  # Assuming the SCP creator is in this module

# Fitness function: Evaluate how well a solution covers the universe
def fitness(solution, subsets, universe):
    covered = set()
    num_subsets_used = sum(solution)  # Count number of subsets used
    for i, included in enumerate(solution):
        if included:
            covered.update(subsets[i])
    
    # Fitness is defined by coverage minus the number of subsets used (penalty)
    return len(covered) - len(universe-covered)*5 - num_subsets_used * 0.5  # Adjust penalty as needed

# Initialize population with random solutions
def initialize_population(pop_size, num_subsets):
    return [[random.choice([0, 1]) for _ in range(num_subsets)] for _ in range(pop_size)]

# Selection function: Select parents based on fitness (tournament selection)
def selection(population, fitness_values):
    selected = []
    for _ in range(len(population)):
        # Randomly select individuals for the tournament
        tournament = random.sample(list(zip(population, fitness_values)), 3)
        # Select the best individual from the tournament
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return random.sample(selected, 2)

# Crossover function: Create new offspring by combining parents
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Mutation function: Randomly flips bits in the solution
def mutate(offspring, mutation_rate=0.003):
    return [(gene if random.random() > mutation_rate else 1 - gene) for gene in offspring]

# Genetic algorithm implementation
def genetic_algorithm(subsets, universe, pop_size=50, generations=50, time_limit=45):
    start_time = time.time()
    population = initialize_population(pop_size, len(subsets))
    best_solution = None
    best_fitness = -float('inf')
    fitness_history = []
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        fitness_values = [fitness(indiv, subsets, universe) for indiv in population]
        fitness_history.append(max(fitness_values))
        
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > best_fitness:
            best_fitness = fitness_values[best_idx]
            best_solution = population[best_idx]
        
        new_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)][:1]
        while len(new_population) < pop_size:
            parent1, parent2 = selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        
        population = new_population[:pop_size]
    
    total_time = time.time() - start_time
    return best_solution, best_fitness, fitness_history, total_time

# Display the output in a format similar to the provided image
def display_output(best_solution, best_fitness, subsets, total_time):
    roll_no = "2022A7PS0009G"  # Assuming the roll number is constant for demonstration purposes
    print(f"Roll no : {roll_no}")
    print(f"Number of subsets in scp_test.json file : {len(subsets)}")
    print("Solution :")
    
    # Displaying the solution in the format "index:value"
    for i, value in enumerate(best_solution):
        print(f"{i}:{value}", end=", " if i % 10 != 9 else "\n")  # Print in blocks of 10 for clarity
    
    print(f"\nFitness value of best state : {int(best_fitness)}")
    print(f"Minimum number of subsets that can cover the Universe-set : {sum(best_solution)}")
    print(f"Time taken : {round(total_time, 2)} seconds")

# Running the genetic algorithm for multiple collection sizes and collecting results
def run_experiment(collection_sizes, num_runs=10, pop_size=50, generations=50, time_limit=45):
    scp = SetCoveringProblemCreator()
    universe = set(range(1, 101))
    all_results = {}

    for size in collection_sizes:
        mean_fitness_over_gens = np.zeros(generations)
        fitnesses = []

        for i in range(num_runs):
            listOfSubsets = scp.Create(100, size)
            best_solution, best_fitness, fitness_history, total_time = genetic_algorithm(listOfSubsets, universe, pop_size, generations, time_limit)
            fitnesses.append(best_fitness)
            mean_fitness_over_gens += np.array(fitness_history)
        
        mean_fitness_over_gens /= num_runs
        all_results[size] = (mean_fitness_over_gens, np.mean(fitnesses), np.std(fitnesses))
    
    return all_results

# Plot mean best fitness value over generations
def plot_mean_fitness_over_generations(all_results):
    plt.figure()
    for size, (mean_fitness_over_gens, _, _) in all_results.items():
        plt.plot(range(len(mean_fitness_over_gens)), mean_fitness_over_gens, label=f'Size={size}')
    plt.xlabel('Generations')
    plt.ylabel('Mean Best Fitness Value')
    plt.title('Mean Best Fitness Value over Generations for Different Collection Sizes')
    plt.legend()
    plt.show()

# Plot mean and standard deviation of best fitness value at the end of 50 generations
def plot_mean_std_fitness_at_end(all_results):
    sizes = []
    means = []
    stds = []
    for size, (_, mean_fitness, std_fitness) in all_results.items():
        sizes.append(size)
        means.append(mean_fitness)
        stds.append(std_fitness)

    plt.figure()
    plt.errorbar(sizes, means, yerr=stds, fmt='-o', capsize=5)
    plt.xlabel('Collection Size')
    plt.ylabel('Mean Best Fitness Value after 50 Generations')
    plt.title('Mean and Standard Deviation of Best Fitness Value at Completion')
    plt.show()

# Running the experiment with display and plots
collection_sizes = [50, 150, 250, 350]  # Sizes to experiment with
all_results = run_experiment(collection_sizes)

# Plotting the required graphs
plot_mean_fitness_over_generations(all_results)
plot_mean_std_fitness_at_end(all_results)

# Run genetic algorithm for specific size and display output
collection_size = 150  # Assuming size is 150 as per the image
scp = SetCoveringProblemCreator()
universe = set(range(1, 101))
listOfSubsets = scp.Create(100, collection_size)

# Run genetic algorithm
best_solution, best_fitness, fitness_history, total_time = genetic_algorithm(listOfSubsets, universe)

# Display the formatted output as shown in the image
display_output(best_solution, best_fitness, listOfSubsets, total_time)
