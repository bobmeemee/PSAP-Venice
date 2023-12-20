import pandas as pd
import random as rd
import numpy as np
import time
from Problem import Movement, time_to_decimal, decimal_to_time, read_data, obj_func, earliest

# ============EVOLUTIONARY ALGORITHM================

# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================
INSTANCE = 1
POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.9
TOURNAMENT_SIZE = 2


def create_population(movements, population_size, deviation_scale=20, time_interval=5):
    # population: [{movement: scheduled_time, movement: scheduled_time, ...}, ...]
    population = []
    for i in range(population_size):
        # create a random solution
        for m in movements:
            time_deviation = round(rd.gauss(0, deviation_scale) / time_interval) * time_interval
            m.set_scheduled_time(time_deviation / 60)
        random_solution = {m: m.get_scheduled_time() for m in movements}
        population.append(random_solution)
    return population


def select_best_parent(population, tournament_size, precedence=None):
    parents = []
    for i in range(tournament_size):
        new_warrior = population[np.random.randint(0, len(population))]
        while new_warrior in parents:
            new_warrior = population[np.random.randint(0, len(population))]
        parents.append(new_warrior)

    parent_fitness = [obj_func(parent, precedence) for parent in parents]
    best_parent = parents[parent_fitness.index(min(parent_fitness))]
    return best_parent


def select_parent_pair(population, tournament_size, precedence=None):
    if len(population) < 2:
        raise ValueError("Population size is too small to select two parents")
    if tournament_size > len(population):
        raise ValueError("Tournament size is too large for the population size")
    if tournament_size < 1:
        raise ValueError("Tournament size must be at least 1")

    if tournament_size == 2 and len(population) == 2:
        return population[0], population[1]

    first_parent = select_best_parent(population, tournament_size, precedence)
    second_parent = select_best_parent(population, tournament_size, precedence)
    while second_parent == first_parent:
        second_parent = select_best_parent(population, tournament_size)
    return first_parent, second_parent


def select_best_parents(population, precedence, mu):
    parents = sorted(population, key=lambda x: obj_func(x, precedence))[:mu]
    return parents


def crossover(parent1, parent2, crossover_probability=0.5, n_points=2, number_of_children=2):
    if crossover_probability <= np.random.rand():
        return parent1, parent2

    if n_points > len(parent1):
        raise ValueError("n_points must be smaller than the length of the parents")

    parent1_keys = list(parent1.keys())
    parent2_keys = list(parent2.keys())

    # select the crossover points
    length = len(parent1_keys)
    rnd = []
    for i in range(n_points):
        r = np.random.randint(0, length)
        while r in rnd:
            r = np.random.randint(0, length)
        rnd.append(r)

    # create the children
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(n_points):
        if i % 2 == 0:
            child1[parent1_keys[rnd[i]]] = parent2[parent2_keys[rnd[i]]]
        else:
            child2[parent2_keys[rnd[i]]] = parent1[parent1_keys[rnd[i]]]

    return child1, child2


def mutation(genome, mutation_probability=0.1):
    for i in range(len(genome)):
        if mutation_probability <= np.random.rand():
            genome[i] += np.random.randint(-1, 1) * 15 / 60  # mutate by 15 minutes
    return genome


# lambda = population size
# mu = number of parents
def solve_EA(movements, precedence, max_time, generations, mu, lmbda, mutation_probability, crossover_probability,
             tournament_size, n_points, deviation_scale, time_interval):
    try:
        assert len(movements) >= lmbda
    except AssertionError:
        print("Lambda must be smaller than the number of movements")
        return None, None

    try:
        assert lmbda % mu == 0
    except AssertionError:
        print("Lambda must be a multiple of mu")
        return None, None

    start_time = time.time()
    # create the initial population
    population = create_population(movements, lmbda, deviation_scale, time_interval)
    initial_best_obj_val = min([obj_func(solution) for solution in population])
    generation = 0
    while time.time() - start_time < max_time and generation < generations:
        # select the parents
        parents = sorted(population, key=lambda x: obj_func(x, precedence))[:mu]
        for idx in range(lmbda // mu):
            # crossover the parents
            child1, child2 = crossover(parents[idx], parents[idx + 1], crossover_probability, n_points)

            # mutate the children
            child1 = mutation(child1, mutation_probability)
            child2 = mutation(child2, mutation_probability)

            # add the children to the population
            population.append(child1)
            population.append(child2)

        # remove the worst solutions from the population
        population = sorted(population, key=lambda x: obj_func(x, precedence))
        population = population[:lmbda]

        # update the best objective value
        best_obj_val = obj_func(population[0], precedence)
        if best_obj_val < initial_best_obj_val:
            initial_best_obj_val = best_obj_val

        generation += 1

        # TODO: time check, if time is up, return the best solution found so far
        # TODO: check if the best solution is valid, if not, return None, None
        # TODO: check if this is correct according to the book
    return population[0], initial_best_obj_val


def solution_generating_procedure(movements: list, l, t, solver=None):
    # movements is a list of movements
    # sort the movements by time
    sorted_movements = sorted(movements, key=lambda x: x.optimal_time)

    # select the first l movements
    movements_subset = sorted_movements[:l]

    fixed_movements = []
    precedence = {}
    while len(fixed_movements) != len(movements):
        # unite the new subset with the fixed movements
        problem_subset = fixed_movements + movements_subset
        # solve the problem with the new subset and the precedence constraints
        solution, obj_val = solver

        # if no solution was found, return None
        if solution is None:
            print("No solution found while using precedence constraints")
            print("reached: ", len(fixed_movements), "/", len(movements))
            return None, None
        else:
            # get the earliest movement in the solution that is not in the fixed movements
            first_movement, first_movement_time = earliest(solution, movements_subset)
            # append the precedence constraints {first_movement: {m_i: time_difference_i}}
            first_movement_precedences = {}
            for m, time in solution.items():
                if m != first_movement:
                    difference = time - first_movement_time
                    first_movement_precedences[m] = difference
            precedence[first_movement] = first_movement_precedences

            # append the movement to the fixed movements
            fixed_movements.append(first_movement)

        # remove the first movement from the candidates for the subset and select the next l earliest movements
        sorted_movements.remove(first_movement)
        movements_subset = sorted_movements[:l]

    if len(fixed_movements) == len(movements):
        return solution, obj_val
    else:
        return None, None


if __name__ == '__main__':
    # read in the data
    df_movimenti, df_precedenze = read_data(INSTANCE)

    initial_solution = create_population(df_movimenti, df_precedenze, population_size=2, deviation_scale=20)

    print(time_to_decimal("00:15:00"))
