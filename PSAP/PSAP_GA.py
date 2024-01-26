import pandas as pd
import random as rd
import numpy as np
import time

from matplotlib import pyplot as plt

from PSAP.Problem import validate_solution, \
     Solver, prepare_movements, SolutionGenerator

# ============EVOLUTIONARY ALGORITHM================

class GeneticAlgorithm(Solver):
    def __init__(self, max_time, time_interval, vessel_time_window, population_size=100, generations=1000,
                 mutation_probability=0.1, mutation_scale=30, crossover_probability=0.5, n_points=2,
                 initial_deviation_scale=20):
        super().__init__(max_time, time_interval, vessel_time_window)
        self.population_size = population_size
        self.generations = generations
        self.mutation_probability = mutation_probability
        self.mutation_scale = mutation_scale
        self.crossover_probability = crossover_probability
        self.n_points = n_points
        self.initial_deviation_scale = initial_deviation_scale

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def select_parent_pair(population, tournament_size=2, precedence=None):
        if len(population) < 2:
            raise ValueError("Population size is too small to select two parents")
        if tournament_size > len(population):
            raise ValueError("Tournament size is too large for the population size")
        if tournament_size < 1:
            raise ValueError("Tournament size must be at least 1")

        if tournament_size == 2 and len(population) == 2:
            return population[0], population[1]

        first_parent = GeneticAlgorithm.select_best_parent(population, tournament_size, precedence)
        second_parent = GeneticAlgorithm.select_best_parent(population, tournament_size, precedence)
        while second_parent == first_parent:
            second_parent = GeneticAlgorithm.select_best_parent(population, tournament_size)
        return first_parent, second_parent

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def mutation(genome, mutation_probability=0.1, mutation_scale=30, time_interval=5):
        for movement in genome.keys():
            if mutation_probability > np.random.rand():
                time_deviation = round(rd.gauss(0, mutation_scale) / time_interval) * time_interval
                genome[movement] += time_deviation / 60
        return genome

    def generate_parameters(self, population_size_rng, generations_rng, mutation_probability_rng,
                            mutation_scale_rng, crossover_probability_rng, n_points_rng, initial_deviation_scale_rng):
        # generate the parameters
        self.population_size = rd.randint(population_size_rng[0], population_size_rng[1])
        self.generations = rd.randint(generations_rng[0], generations_rng[1])
        self.mutation_probability = rd.uniform(mutation_probability_rng[0], mutation_probability_rng[1])
        self.mutation_scale = rd.randint(mutation_scale_rng[0], mutation_scale_rng[1])
        self.crossover_probability = rd.uniform(crossover_probability_rng[0], crossover_probability_rng[1])
        self.n_points = rd.randint(n_points_rng[0], n_points_rng[1])
        self.initial_deviation_scale = rd.randint(initial_deviation_scale_rng[0], initial_deviation_scale_rng[1])

        # make sure that deviation is in intervals of 5
        self.initial_deviation_scale = round(self.initial_deviation_scale / 5) * 5
        self.mutation_scale = round(self.mutation_scale / 5) * 5

        return (self.population_size, self.generations, self.mutation_probability, self.mutation_scale,
                self.crossover_probability, self.n_points, self.initial_deviation_scale)

    def solve(self, movements, precedence=None):
        start_time = time.time()
        # create the initial population

        population = self.create_population(movements, self.population_size, self.initial_deviation_scale,
                                            self.time_interval)
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
        while time.time() - start_time < self.max_time and generation < self.generations:

            for idx in range(population_size // 2):
                # select the parents
                parent1, parent2 = self.select_parent_pair(parents, 2, precedence)
                child1, child2 = self.crossover2(parent1, parent2, self.crossover_probability, self.n_points)

                # mutate the children
                child1 = self.mutation(child1, self.mutation_probability, self.mutation_scale)
                child2 = self.mutation(child2, self.mutation_probability, self.mutation_scale)

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


if __name__ == '__main__':
    sol_found = False
    instance = 1
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'epochs', 'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window'])

    time_window = 60 * 6
    time_interval = 5

    solver = GeneticAlgorithm(max_time=60 * 60 * 2, time_interval=time_interval, vessel_time_window=time_window,
                              population_size=300, generations=100, mutation_probability=0.2, mutation_scale=40,
                              crossover_probability=0.7, n_points=2, initial_deviation_scale=15)

    solutionGenerator = SolutionGenerator(movements=[], l=3, t=5, solver=solver, time_interval=5,
                                          vessel_time_window=60 * 6)
    while instance < 11:
        print("=====================================")
        print("Instance: ", instance)

        # read in the data
        result_list = prepare_movements(instance)
        result_dict = {m: m.optimal_time for m in result_list}
        print("Objective value initial solution: ", obj_func(result_dict))

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

            # generate the parameters
            population_size, generations, mutation_probability, mutation_scale, crossover_probability, \
                n_points, initial_deviation_scale = solver.generate_parameters(population_size_rng=[100, 300],
                                                                               generations_rng=[100, 300],
                                                                               mutation_probability_rng=[0.1, 0.3],
                                                                               mutation_scale_rng=[20, 60],
                                                                               crossover_probability_rng=[0.5, 0.8],
                                                                               n_points_rng=[1, 3],
                                                                               initial_deviation_scale_rng=[10, 30])

            solutionGenerator.set_movements(result_list.copy())
            initial_solution, obj_val, prev_initial_solution = solutionGenerator.generate_solution()

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
