import pandas as pd
import random as rd
import numpy as np
import time

from matplotlib import pyplot as plt

from Problem import Movement, time_to_decimal, decimal_to_time, read_data, earliest, validate_solution, \
    generate_initial_solution

# ============EVOLUTIONARY ALGORITHM================

# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================
# INSTANCE = 1
TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6


def create_population(movements, population_size, deviation_scale=20, time_interval=5):
    # population: [{movement: scheduled_time, movement: scheduled_time, ...}, ...]
    population = []
    for i in range(population_size):
        # create a random solution
        for m in movements:
            time_deviation = round(rd.gauss(0, deviation_scale) / time_interval) * time_interval
            m.set_scheduled_time(m.optimal_time + time_deviation / 60)
        random_solution = {m: m.get_scheduled_time() for m in movements}
        population.append(random_solution)
    return population


def select_best_parent(population, tournament_size=2, precedence=None):
    parents = []
    for i in range(tournament_size):
        new_warrior = population[np.random.randint(0, len(population))]
        while new_warrior in parents:
            new_warrior = population[np.random.randint(0, len(population))]
        parents.append(new_warrior)

    parent_fitness = [obj_func(parent, precedence) for parent in parents]
    best_parent = parents[parent_fitness.index(min(parent_fitness))]
    return best_parent


def select_parent_pair(population, tournament_size=2, precedence=None):
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


def crossover(parent1, parent2, crossover_probability=0.5, n_points=2):
    if crossover_probability <= np.random.rand():
        return parent1, parent2

    if n_points > len(parent1):
        raise ValueError("n_points must be smaller than the length of the parents")

    keys = list(parent1.keys())

    # select the crossover points
    length = len(keys)
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
            child1[keys[rnd[i]]] = parent2[keys[rnd[i]]]
        else:
            child2[keys[rnd[i]]] = parent1[keys[rnd[i]]]

    return child1, child2


def crossover2(parent1, parent2, crossover_probability=0.5, n_points=2):
    if crossover_probability <= np.random.rand():
        return parent1, parent2

    if n_points > len(parent1):
        raise ValueError("n_points must be smaller than the length of the parents")

    keys = list(parent1.keys())

    # select the crossover points
    length = len(keys)
    rnd = []
    for i in range(n_points):
        r = np.random.randint(0, length)
        while r in rnd:
            r = np.random.randint(0, length)
        rnd.append(r)

    # create the children
    child1 = parent1.copy()
    child2 = parent2.copy()

    child1_changing = True
    for i in range(n_points):
        if i in rnd:
            child1_changing = not child1_changing
        if child1_changing:
            child1[keys[i]] = parent2[keys[i]]
        else:
            child2[keys[i]] = parent1[keys[i]]

    return child1, child2


def mutation(genome, mutation_probability=0.1, mutation_scale=30, time_interval=5):
    for movement in genome.keys():
        if mutation_probability > np.random.rand():
            time_deviation = round(rd.gauss(0, mutation_scale) / time_interval) * time_interval
            genome[movement] += time_deviation / 60
    return genome


def obj_func(solution, precedence=None):
    cost = 0
    for key, value in solution.items():
        # penalty for deviation from optimal time
        if key.vessel_type == 'Cargo ship':
            cost += 1 * abs(key.optimal_time - value) * 1
        else:
            cost += 1 * abs(key.optimal_time - value) * 1

        # if abs(key.optimal_time - value) > TIME_WINDOW / 60 / 2:
        #     cost += 100 * abs(key.optimal_time - value)

        # penalty for headway violations
        for key2, value2 in solution.items():
            if key != key2:
                if key.headway.get(key2.id_number)[0] == 0:
                    # m and m' are the same vessel, m can't be scheduled before m'
                    delta_t = value2 - value
                    if delta_t < 0:
                        cost += 1 * abs(delta_t)
                elif key.headway.get(key2.id_number)[0] == 1:
                    # headway has to be applied
                    delta_t = value2 - value
                    if delta_t < key.headway.get(key2.id_number)[1] and delta_t > 0.:
                        cost += abs(delta_t) * 180
                else:
                    # no headway has to be applied
                    cost += 0

        # check if the precedence constraints are satisfied
        if precedence is not None:
            precedences = precedence.get(key)  # {m': headway}
            if precedences is None:
                continue
            for other_movement, headway in precedences.items():
                if key == other_movement:
                    print('ERROR: Movement is in its own precedence constraints')
                    return False

                # precedence can be negative
                time_difference = solution[other_movement] - value
                if headway >= 0:
                    if time_difference < headway:
                        cost += 1 * abs(time_difference - headway)

                else:
                    if time_difference > headway:
                        cost += 1 * abs(time_difference - headway)

    return cost


# basic GA
def solve_GA(movements, precedence, max_time, population_size, generations, mutation_probability, mutation_scale,
             crossover_probability, n_points, initial_deviation_scale, time_interval, vessel_time_window):
    start_time = time.time()
    # create the initial population

    population = create_population(movements, population_size, initial_deviation_scale, time_interval)
    initial_best_obj_val = min([obj_func(solution, precedence) for solution in population])
    print("Initial best obj val: ", initial_best_obj_val)
    parents = sorted(population, key=lambda x: obj_func(x, precedence))

    current_best_solution = parents[0]
    if validate_solution(current_best_solution, vessel_time_window, print_errors=False):
        print("Initial solution is valid")
        return current_best_solution, initial_best_obj_val

    generation = 0
    obj_values = [initial_best_obj_val]
    children = []
    child_one_obj_values = []
    while time.time() - start_time < max_time and generation < generations:

        for idx in range(population_size // 2):
            # select the parents
            parent1, parent2 = select_parent_pair(parents, 2, precedence)
            child1, child2 = crossover2(parent1, parent2, crossover_probability, n_points)

            # mutate the children
            child1 = mutation(child1, mutation_probability, mutation_scale)
            child2 = mutation(child2, mutation_probability, mutation_scale)

            children.append(child1)
            children.append(child2)

            # child_one_obj_values.append(obj_func(child1, precedence))

            # check if the child is better than the best solution
            # if so, print
            if obj_func(child1, precedence) < initial_best_obj_val:
                print("child1 is better than best solution")

        parents = children.copy()
        children = []

        # update the best objective value
        best_obj_val = min([obj_func(solution, precedence) for solution in parents])
        if best_obj_val < initial_best_obj_val:
            initial_best_obj_val = best_obj_val
            # update the best solution, the list is not sorted
            for solution in parents:
                if obj_func(solution, precedence) == best_obj_val:
                    current_best_solution = solution
                    break

        generation += 1
        obj_values.append(best_obj_val)
        print("generation: ", generation)

    print("final generation: ", generation)
    # plot the objective values log scale

    x_ax = np.linspace(0, len(child_one_obj_values), len(obj_values))
    plt.plot(obj_values, color='blue')
    # plt.plot(child_one_obj_values, color='red')

    plt.show()

    if validate_solution(current_best_solution, vessel_time_window, print_errors=True):
        return current_best_solution, initial_best_obj_val
    else:
        print("Final solution is not valid")
        print("obj_val: ", initial_best_obj_val)
        return None, None


def solution_generating_procedure(movements: list, l, t, generations=1000, population_size=100,
                                  initial_deviation_scale=20,
                                  mutation_probability=0.1, mutation_scale=30, crossover_probability=0.9, n_points=2,
                                  time_interval=5, vessel_time_window=360):
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
        solution, obj_val = solve_GA(problem_subset, precedence=precedence, max_time=t, generations=generations,
                                     population_size=population_size, mutation_probability=mutation_probability,
                                     mutation_scale=mutation_scale, crossover_probability=crossover_probability,
                                     n_points=n_points, initial_deviation_scale=initial_deviation_scale,
                                     time_interval=time_interval, vessel_time_window=vessel_time_window)

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


def generate_parameters():
    pass


if __name__ == '__main__':
    sol_found = False
    instance = 10
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'epochs', 'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window'])

    time_window = 60 * 6
    time_interval = 5
    while instance < 11:
        print("=====================================")
        print("Instance: ", instance)
        # read in the data
        df_movimenti, df_precedenze = read_data(1)
        # generate the initial solution
        initial_solution = generate_initial_solution(df=df_movimenti, df_headway=df_precedenze, deviation_scale=1,
                                                     time_interval=TIME_INTERVAL)
        movements = list(initial_solution.keys())
        sorted_movements = sorted(movements, key=lambda x: x.optimal_time)
        result_list = [elem for index, elem in enumerate(sorted_movements, 1) if index % 2 != 0]

        print("Objective value initial solution: ", obj_func(initial_solution))
        generations = 100
        population_size = 300
        mutation_probability = .2
        mutation_scale = 40
        crossover_probability = .7
        n_points = 2
        initial_deviation_scale = 15
        time_interval = 5
        vessel_time_window = 360
        # run the solution generating procedure 10 times for each instance and save the results
        # TODO: make this work
        for _ in range(1):
            initial_solution, obj_val = solution_generating_procedure(result_list, 2, 10,
                                                                      generations=generations,
                                                                      population_size=population_size,
                                                                      mutation_probability=mutation_probability,
                                                                      mutation_scale=mutation_scale,
                                                                      crossover_probability=crossover_probability,
                                                                      n_points=n_points,
                                                                      initial_deviation_scale=initial_deviation_scale,
                                                                      time_interval=time_interval,
                                                                      vessel_time_window=vessel_time_window)

            if initial_solution is not None:
                # set the movement scheduled to the result of the solution generating procedure
                for m, t in initial_solution.items():
                    m.set_scheduled_time(t)
                print("Solution", _, " found for instance", instance, "(", len(initial_solution), ")")
                obj_val = obj_func(initial_solution)
                print("Objective value: ", obj_val)

        instance += 1

        try:
            df.to_excel('results/EA/output.xlsx', index=False)
        except PermissionError:
            print("Please close the file output.xlsx and try again")
        except FileNotFoundError:
            print("File not found")

        print("solutions found: ", len(df.index))
