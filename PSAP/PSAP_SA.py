import pandas as pd
import time
import random as rd
import numpy as np
import os

from PSAP.Problem import (validate_solution, generate_neighbor_solution,
                          obj_func, earliest, Solver, SolutionGenerator, prepare_movements)


class SimulatedAnnealing(Solver):
    def __init__(self, max_time, epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements,
                 lengthMaxMov,
                 time_interval, vessel_time_window):
        super().__init__(max_time, time_interval, vessel_time_window)
        self.epochs = epochs
        self.neighborhood_size = neighborhood_size
        self.t0 = t0
        self.alpha = alpha
        self.neighbor_deviation_scale = neighbor_deviation_scale
        self.affected_movements = affected_movements
        self.lengthMaxMov = lengthMaxMov

    def generate_parameters(self, epochs_rng, neighborhood_size_rng, t0_rng, alpha_rng, neighbor_deviation_scale_rng,
                            affected_movements_rng):
        self.epochs = rd.randint(epochs_rng[0], epochs_rng[1])
        self.neighborhood_size = rd.randint(neighborhood_size_rng[0], neighborhood_size_rng[1])
        self.t0 = rd.randint(t0_rng[0], t0_rng[1])
        self.alpha = rd.uniform(alpha_rng[0], alpha_rng[1])
        self.neighbor_deviation_scale = rd.randint(neighbor_deviation_scale_rng[0], neighbor_deviation_scale_rng[1])
        self.affected_movements = rd.randint(affected_movements_rng[0], affected_movements_rng[1])

        return (self.epochs, self.neighborhood_size, self.t0, self.alpha, self.neighbor_deviation_scale,
                self.affected_movements)

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_neighborhood_size(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

    def set_t0(self, t0):
        self.t0 = t0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_neighbor_deviation_scale(self, neighbor_deviation_scale):
        self.neighbor_deviation_scale = neighbor_deviation_scale

    def set_affected_movements(self, affected_movements):
        self.affected_movements = affected_movements

    def set_lengthMaxMov(self, lengthMaxMov):
        self.lengthMaxMov = lengthMaxMov

    def solve(self, movements, precedence=None):
        # the initial solution is a dictionary with movements as keys and scheduled times as values
        # the precedence constraints are a dictionary with movements as keys and lists of movements as values
        # the max_time is the maximum time in minutes that the movements can be scheduled

        # start the timer
        start_time = time.time()
        initial_solution = {m: m.optimal_time for m in movements}
        initial_obj_val = obj_func(initial_solution)
        e = 0
        attempt = 0
        stopping_condition_counter = 0

        # while the time limit is not reached
        while time.time() - start_time < self.max_time:
            # make sure the algorithm does't stop at the first valid solution when it reaches the max amount of movements
            if validate_solution(initial_solution, self.vessel_time_window) and len(movements) != self.lengthMaxMov:
                return initial_solution, initial_obj_val

            if e >= self.epochs:
                initial_solution = {m: m.optimal_time for m in movements}
                initial_obj_val = obj_func(initial_solution)
                e = 0
                attempt += 1

            for i in range(self.neighborhood_size):
                new_solution = initial_solution.copy()
                new_solution = generate_neighbor_solution(new_solution, affected_movements=self.affected_movements,
                                                          deviation_scale=self.neighbor_deviation_scale,
                                                          time_interval=self.time_interval)

                new_obj_val = obj_func(new_solution, precedence=precedence)

                # if the new solution is better, accept it
                if new_obj_val < initial_obj_val:
                    initial_solution = new_solution.copy()
                    initial_obj_val = new_obj_val
                else:
                    # if the new solution is worse, accept it with a probability
                    p = rd.random()
                    try:
                        if p < np.exp(-(new_obj_val - initial_obj_val) / self.t0):
                            initial_solution = new_solution.copy()
                            initial_obj_val = new_obj_val
                    except ZeroDivisionError:
                        # never accept a worse solution if the temperature is 0
                        pass

                if abs(new_obj_val - initial_obj_val) < 0.000001:
                    stopping_condition_counter += 1
                    if stopping_condition_counter == 10 * self.neighborhood_size:
                        initial_solution = {m: m.optimal_time for m in movements}
                        initial_obj_val = obj_func(initial_solution)
                        e = 0
                        attempt += 1

            # update the temperature
            self.t0 *= self.alpha
            e += 1

        if validate_solution(initial_solution, self.vessel_time_window, print_errors=True):
            return initial_solution, initial_obj_val
        else:
            return None, None

    def __str__(self):
        return "Simulated Annealing" + "\n" + \
            "t0: " + str(self.t0) + "\n" + \
            "alpha: " + str(self.alpha) + "\n"

"""
# ============SIMULATED ANNEALING================
# method one: generate a random solution and improve on it by also respecting the precedence constraints
def solve_with_precedence_constraints_SA(movements: list, precedence: dict, max_time: int, lengthMaxMov,
                                         epochs, neighborhood_size: int,
                                         t0=100, alpha=0.98, neighbor_deviation_scale=40, affected_movements=3,
                                         time_interval=5, vessel_time_window=60 * 6):
    # the initial solution is a dictionary with movements as keys and scheduled times as values
    # the precedence constraints are a dictionary with movements as keys and lists of movements as values
    # the max_time is the maximum time in minutes that the movements can be scheduled

    # start the timer
    start_time = time.time()
    initial_solution = {m: m.optimal_time for m in movements}
    initial_obj_val = obj_func(initial_solution)
    e = 0
    attempt = 0
    stopping_condition_counter = 0

    # while the time limit is not reached
    while time.time() - start_time < max_time:
        # make sure the algorithm does't stop at the first valid solution when it reaches the max amount of movements
        if validate_solution(initial_solution, vessel_time_window) and len(movements) != lengthMaxMov:
            return initial_solution, initial_obj_val

        if e >= epochs:
            initial_solution = {m: m.optimal_time for m in movements}
            initial_obj_val = obj_func(initial_solution)
            e = 0
            attempt += 1

        for i in range(neighborhood_size):
            new_solution = initial_solution.copy()
            new_solution = generate_neighbor_solution(new_solution, affected_movements=affected_movements,
                                                      deviation_scale=neighbor_deviation_scale,
                                                      time_interval=time_interval)

            new_obj_val = obj_func(new_solution, precedence=precedence)

            # if the new solution is better, accept it
            if new_obj_val < initial_obj_val:
                initial_solution = new_solution.copy()
                initial_obj_val = new_obj_val
            else:
                # if the new solution is worse, accept it with a probability
                p = rd.random()
                try:
                    if p < np.exp(-(new_obj_val - initial_obj_val) / t0):
                        initial_solution = new_solution.copy()
                        initial_obj_val = new_obj_val
                except ZeroDivisionError:
                    # never accept a worse solution if the temperature is 0
                    pass

            if abs(new_obj_val - initial_obj_val) < 0.000001:
                stopping_condition_counter += 1
                if stopping_condition_counter == 10 * neighborhood_size:
                    initial_solution = {m: m.optimal_time for m in movements}
                    initial_obj_val = obj_func(initial_solution)
                    e = 0
                    attempt += 1

        # update the temperature
        t0 = alpha * t0
        e += 1

    if validate_solution(initial_solution, vessel_time_window, print_errors=True):
        return initial_solution, initial_obj_val
    else:
        return None, None


def solution_generating_procedure(movements: list, l, t, epochs=200, neighborhood_size=4, t0=100,
                                  lengthMaxMov=0, do_precedence=True,
                                  alpha=.8, neighbor_deviation_scale=40, affected_movements=3, time_interval=5,
                                  vessel_time_window=60 * 6):
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
        if not do_precedence:
            precedence = {}

        solution, obj_val = solve_with_precedence_constraints_SA(problem_subset, precedence, max_time=t,
                                                                 epochs=epochs,
                                                                 lengthMaxMov=lengthMaxMov,
                                                                 neighborhood_size=neighborhood_size, t0=t0,
                                                                 alpha=alpha,
                                                                 neighbor_deviation_scale=neighbor_deviation_scale,
                                                                 affected_movements=affected_movements,
                                                                 time_interval=time_interval,
                                                                 vessel_time_window=vessel_time_window)

        # if no solution was found, return None
        if solution is None:
            print("No solution found while using precedence constraints")
            print("reached: ", len(fixed_movements), "/", len(movements))
            return None, prev_obj_val, prev_solution
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
        prev_solution = solution
        prev_obj_val = obj_val

    return solution, obj_val, None
"""

def generate_parameters(epochs_rng, neighborhood_size_rng, t0_rng, alpha_rng, neighbor_deviation_scale_rng,
                        affected_movements_rng):
    epochs = rd.randint(epochs_rng[0], epochs_rng[1])
    neighborhood_size = rd.randint(neighborhood_size_rng[0], neighborhood_size_rng[1])
    t0 = rd.randint(t0_rng[0], t0_rng[1])
    alpha = rd.uniform(alpha_rng[0], alpha_rng[1])
    neighbor_deviation_scale = rd.randint(neighbor_deviation_scale_rng[0], neighbor_deviation_scale_rng[1])
    affected_movements = rd.randint(affected_movements_rng[0], affected_movements_rng[1])

    return epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements


if __name__ == '__main__':
    sol_found = False
    instance = 1
    valid_solutions = []
    objective_values = []

    project_root = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join('../results/SA/ignore.xlsx')

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'epochs', 'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window', 'solution_found'])

    df_instance = pd.DataFrame(columns=['instance', 'number_of_movements', 'number_of_vessels', 'average_headway',
                                        'std_dev_headway', 'spread', 'average_time_between_movements',
                                        'average_travel_time'])

    time_window = 60 * 6
    time_interval = 5

    solver = SimulatedAnnealing(max_time=60, epochs=100, neighborhood_size=4, t0=100, alpha=0.98,
                                neighbor_deviation_scale=40, affected_movements=3, lengthMaxMov=0,
                                time_interval=5, vessel_time_window=60 * 6)

    solutionGenerator = SolutionGenerator(movements=[], l=3, t=5, solver=solver, time_interval=5,
                                          vessel_time_window=60 * 6)

    while instance < 101:
        print("=====================================")
        print("Instance: ", instance)

        # read in the data
        # return a list with half of the movements and all their characteristics
        result_list = prepare_movements(instance)
        lengthMaxMov = len(result_list)

        # to dict for obj_func
        result_dict = {m: m.optimal_time for m in result_list}
        print("Objective value initial solution: ", obj_func(result_dict))

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(1):

            # generate random parameters
            epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements = (
                solver.generate_parameters(
                    epochs_rng=[100, 100],
                    neighborhood_size_rng=[4, 4],
                    t0_rng=[40, 500],
                    alpha_rng=[0.5, 0.99],
                    neighbor_deviation_scale_rng=[40, 40],
                    affected_movements_rng=[4, 4]))

            # solve the problem with the generated parameters
            solutionGenerator.set_movements(result_list.copy())
            initial_solution, obj_val, prev_initial_solution = solutionGenerator.generate_solution()

            if initial_solution is not None:
                # set the movement scheduled to the result of the solution generating procedure
                for m, t in initial_solution.items():
                    m.set_scheduled_time(t)
                print("Solution", _, " found for instance", instance, "(", len(initial_solution), ")")
                obj_val = obj_func(initial_solution)
                print("Objective value: ", obj_val)
                df.loc[len(df.index)] = [instance, len(initial_solution),
                                         np.median([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                                         affected_movements, time_interval, time_window, 1]

            else:
                print("No solution found for instance", instance)
                for m, t in prev_initial_solution.items():
                    m.set_scheduled_time(t)
                obj_val = obj_func(prev_initial_solution)

                df.loc[len(df.index)] = [instance, len(prev_initial_solution),
                                         np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                                         affected_movements, time_interval, time_window, 0]

        instance += 1

        try:
            df.to_excel(save_path, index=False)
        except PermissionError:
            print("Please close the file output.xlsx and try again")

        print("solutions found: ", len(df.index))
