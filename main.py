from Optimizers.Population import Population
from Optimizers.Algorithms import *
import numpy as np

def sphere(x):
    return np.sum(x**2)

lower_bound = -100
upper_bound = 100
size = 3
dimension = 4
max_iter = size

# Create and initialize population
population = Population(lower_bound = lower_bound, upper_bound = upper_bound,
                        size=size, dimension=dimension, objective_function=sphere)
population.initialize()

# Get population
populations = population.population

print(f"population:\n{populations}\n")

# Get evaluate value of the population
eval_value = population.eval_value

# Initialize algorithm
jaya = Jaya(dimension, max_iter)

for iter in range(max_iter):

    current_agent = populations[iter]

    jaya.set_agents(current_agent, populations[np.argmin(np.min(eval_value))], populations[np.argmax(np.max(eval_value))], 
                    populations[np.argmin(np.min(eval_value))])

    updated_pop = jaya.step(iter)

    updated_eval_value = population.updated_eval(updated_pop)

    print(f"\n=================================================================\nNumber of iteration: {iter}\n=================================================================")
    print(f"jaya ==>> current_agent: {current_agent}\n")
    print(f"jaya ==>> updated_egent: {updated_pop}")

    '''
    # Random Print
    j_current_agent = jaya.current_agent
    j_local_optimum_agent = jaya.local_optimum_agent
    j_global_optimum_agent = jaya.global_optimum_agent
    print(f"jaya ==>> jcurrent_agent: {j_current_agent}\n")
    print(f"population ==>> eval_value: {eval_value}\n")
    print(f"jaya ==>> local_optimum_agent: {j_local_optimum_agent}\n")
    print(f"jaya ==>> global_optimum_agent: {j_global_optimum_agent}\n")
    print(f"\njava ==>> updated_eval_value: {updated_eval_value}")
    '''