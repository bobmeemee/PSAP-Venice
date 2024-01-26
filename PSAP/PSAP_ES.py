import pandas as pd
import random as rd
import numpy as np
import time

from matplotlib import pyplot as plt

from PSAP.Problem import validate_solution, Solver, prepare_movements, SolutionGenerator


# ============EVOLUTIONARY ALGORITHM================

# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================

# lambda = number of children
# mu = number of parents
# mu + lambda = population size
class EvolutionaryStrategy(Solver):
    def __init__(self, max_time, time_interval, vessel_time_window, generations=1000, mu=10, lmbda=100, deviation_scale=20,
                 mutation_probability=0.1, mutation_scale=30):
        super().__init__(max_time, time_interval, vessel_time_window)
        self.generations = generations
        self.mu = mu
        self.lmbda = lmbda
        self.mutation_probability = mutation_probability
        self.mutation_scale = mutation_scale
        self.deviation_scale = deviation_scale

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
    def mutation(genome, mutation_probability=0.1, mutation_scale=30, time_interval=5):
        for movement in genome.keys():
            if mutation_probability > np.random.rand():
                time_deviation = round(rd.gauss(0, mutation_scale) / time_interval) * time_interval
                genome[movement] += time_deviation / 60
        return genome

    def generate_parameters(self, mu_rng, fact, mutation_probability_rng, mutation_scale_rng, deviation_scale_rng):
        # generate the parameters
        self.mu = rd.randint(mu_rng[0], mu_rng[1])
        self.lmbda = self.mu * fact

        self.mutation_probability = rd.uniform(mutation_probability_rng[0], mutation_probability_rng[1])
        self.mutation_scale = rd.randint(mutation_scale_rng[0], mutation_scale_rng[1])
        self.deviation_scale = rd.randint(deviation_scale_rng[0], deviation_scale_rng[1])

        # make sure that deviation is in intervals of 5
        self.mutation_scale = round(self.mutation_scale / 5) * 5

        return self.mutation_probability, self.mutation_scale, self.deviation_scale, self.mu, self.lmbda

    def solve(self, movements, precedence=None):
        try:
            assert self.lmbda % self.mu == 0
        except AssertionError:
            print("Lambda must be a multiple of mu")
            return None, None

        start_time = time.time()
        # create the initial population

        population = self.create_population(movements, self.lmbda, self.deviation_scale, self.time_interval)
        initial_best_obj_val = min([obj_func(solution, precedence) for solution in population])
        print("Initial best obj val: ", initial_best_obj_val)
        parents = sorted(population, key=lambda x: obj_func(x, precedence))

        current_best_solution = parents[0]
        if validate_solution(current_best_solution, self.vessel_time_window, print_errors=False):
            print("Initial solution is valid")
            return current_best_solution, initial_best_obj_val

        generation = 0
        obj_values = []
        children = []
        child_one_obj_values = []
        while time.time() - start_time < self.max_time and generation < self.generations:

            for idx in range(self.lmbda // self.mu):

                # mutate the children
                child1 = self.mutation(parents[idx], self.mutation_probability, self.mutation_scale)

                parents.append(child1)
                child_one_obj_values.append(obj_func(child1, precedence))

                # check if the child is better than the best solution
                # if so, print
                if obj_func(child1, precedence) < initial_best_obj_val:
                    print("child1 is better than best solution")
                    print(idx, " ", generation)

            parents += children

            # remove the worst solutions from the population and keep the best mu solutions
            parents = sorted(parents, key=lambda x: obj_func(x, precedence))[:self.mu]
            # update the best objective value
            best_obj_val = obj_func(parents[0], precedence)
            if best_obj_val < initial_best_obj_val:
                initial_best_obj_val = best_obj_val
                current_best_solution = parents[0]

            generation += 1
            obj_values.append(best_obj_val)
        print("final generation: ", generation)
        # plot the objective values log scale

        x_ax = np.linspace(0, len(child_one_obj_values), len(obj_values))
        plt.plot(x_ax, obj_values, color='blue')
        plt.plot(child_one_obj_values, color='red')

        plt.show()

        if validate_solution(current_best_solution, self.vessel_time_window, print_errors=True):
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
            cost += 1 * abs(key.optimal_time - value) * 5
        else:
            cost += 1 * abs(key.optimal_time - value) * 5

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
                        cost += abs(delta_t) * 2000
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
                        cost += 10 * abs(time_difference - headway)

                else:
                    if time_difference > headway:
                        cost += 10 * abs(time_difference - headway)

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
    while instance < 2:
        print("=====================================")
        print("Instance: ", instance)

        # read in the data
        result_list = prepare_movements(instance)
        result_dict = {m: m.optimal_time for m in result_list}
        print("Objective value initial solution: ", obj_func(result_dict))

        generations = 200
        mu = 2  # number of parents
        lmbda = 2000  # number of children
        mutation_probability = .2
        mutation_scale = 40
        time_interval = 5
        vessel_time_window = 360

        solver = EvolutionaryStrategy(max_time=60 * 60, time_interval=time_interval, vessel_time_window=time_window,
                                      generations=generations, mu=mu, lmbda=lmbda,
                                      mutation_probability=mutation_probability, mutation_scale=mutation_scale)

        solutionGenerator = SolutionGenerator(movements=[], l=3, t=5, solver=solver, time_interval=5,
                                              vessel_time_window=60 * 6)

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(1):

            solutionGenerator.set_movements(result_list.copy())
            initial_solution, obj_val, prev_solution = solutionGenerator.generate_solution()

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
