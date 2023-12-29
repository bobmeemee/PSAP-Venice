import pandas as pd
import time
import random as rd
import numpy as np

from Problem import Movement, time_to_decimal, decimal_to_time, validate_solution, generate_initial_solution, \
    read_data, generate_neighbor_solution, obj_func, validate_precedence_constraints, earliest, collect_instance_data

TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6


class Particle:
    def __init__(self, position=None):
        self.position = position  # dictionary with movements as keys and scheduled times as values
        self.best_position = position  # dictionary with movements as keys and scheduled times as values of the best
        # solution found by the particle)

        self.informants = []
        self.best_informant = None

    def set_informants(self, swarm, percentage):
        self.informants = [self]
        while len(self.informants) < int(percentage * len(swarm)):
            next_informant = swarm[rd.randint(0, len(swarm) - 1)]
            while next_informant in self.informants:
                next_informant = swarm[rd.randint(0, len(swarm) - 1)]
            self.informants.append(next_informant)

        self.best_informant = self.informants[np.argmin([obj_func(solution) for solution in self.informants])]

    def get_previous_best_informant(self):
        return self.best_informant

    def set_best_informant(self):
        self.best_informant = self.informants[np.argmin([obj_func(solution) for solution in self.informants])]


def create_swarm(movements, swarm_size, deviation_scale=20, time_interval=5):
    swarm = []
    for i in range(swarm_size):
        # create a random solution
        for m in movements:
            time_deviation = round(rd.gauss(0, deviation_scale) / time_interval) * time_interval
            m.set_scheduled_time(m.optimal_time + time_deviation / 60)
        random_solution = {m: m.get_scheduled_time() for m in movements}

        # create a particle with the random solution as position
        particle = Particle(position=random_solution.copy())
        swarm.append(particle)
    return swarm


# ============PARITCLE SWARM OPTIMIZATION================
# MILP SOLVER BASED ON PSO
def solve_with_precedence_constraints_PSO(movements: list, precedence: dict, max_time: int, epochs, swarm_size=10,
                                          informant_percentage=0.5,
                                          alpha=0.95, beta=0.8, gamma=0.8, delta=0.8, epsilon=0.8,
                                          vessel_time_window=60 * 6):
    # the initial solution is a dictionary with movements as keys and scheduled times as values
    # the precedence constraints are a dictionary with movements as keys and lists of movements as values
    # the max_time is the maximum time in minutes that the movements can be scheduled

    # start the timer
    start_time = time.time()

    # create the swarm
    swarm = create_swarm(movements, swarm_size)
    all_best_obj_val = min([obj_func(solution, precedence) for solution in swarm])
    all_best_solution = swarm[np.argmin([obj_func(solution, precedence) for solution in swarm])]
    print("Initial best obj val: ", all_best_obj_val)

    e = 0
    attempt = 0
    stopping_condition_counter = 0

    # while the time limit is not reached
    while time.time() - start_time < max_time and e < epochs:
        if e >= epochs:
            initial_solution = {m: m.optimal_time for m in movements}
            initial_obj_val = obj_func(initial_solution)
            e = 0
            attempt += 1

        # get the current best particle and update the all time best particle
        current_best_solution = swarm[np.argmin([obj_func(solution, precedence) for solution in swarm])]
        if obj_func(current_best_solution, precedence) < all_best_obj_val:
            all_best_obj_val = obj_func(current_best_solution, precedence)
            all_best_solution = current_best_solution.copy()

        for particle in swarm:
            # the best solution currently found by the particle
            x = particle.copy()
            # select a small subset of the swarm including the current particle to be the informants
            informants = [particle]
            while len(informants) < informant_percentage * swarm_size:
                next_informant = swarm[rd.randint(0, swarm_size - 1)]
                while next_informant in informants:
                    next_informant = swarm[rd.randint(0, swarm_size - 1)]
                informants.append(next_informant)
            # select the best informant
            x_plus = informants[np.argmin([obj_func(solution, precedence) for solution in informants])]
            # the best solution found by the particle over all epochs
            x_exclamation = all_best_solution.copy()

            # select the best informant
            best_informant = informants[np.argmin([obj_func(solution, precedence) for solution in informants])]

    if validate_solution(initial_solution, vessel_time_window, print_errors=True):
        return initial_solution, initial_obj_val
    else:
        return None, None


def solution_generating_procedure(movements: list, l, t, epochs=200, neighborhood_size=4, t0=100,
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
        solution, obj_val = solve_with_precedence_constraints_PSO(problem_subset, precedence, max_time=t,
                                                                  epochs=epochs)

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


def generate_parameters(epochs_rng, neighborhood_size_rng, t0_rng, alpha_rng, neighbor_deviation_scale_rng,
                        affected_movements_rng):
    pass


if __name__ == '__main__':
    sol_found = False
    instance = 82
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'epochs', 'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window'])

    df_instance = pd.DataFrame(columns=['instance', 'number_of_movements', 'number_of_vessels', 'average_headway',
                                        'std_dev_headway', 'spread', 'average_time_between_movements',
                                        'average_travel_time'])

    time_window = 60 * 6
    time_interval = 5
    while instance < 101:
        print("=====================================")
        print("Instance: ", instance)
        # read in the data
        df_movimenti, df_precedenze, df_tempi = read_data(instance)
        # generate the initial solution
        initial_solution = generate_initial_solution(df=df_movimenti, df_headway=df_precedenze, deviation_scale=1,
                                                     time_interval=TIME_INTERVAL, df_tempi=df_tempi)
        movements = list(initial_solution.keys())
        sorted_movements = sorted(movements, key=lambda x: x.optimal_time)
        result_list = [elem for index, elem in enumerate(sorted_movements, 1) if index % 2 != 0]
        print("Objective value initial solution: ", obj_func(initial_solution))

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(10):

            initial_solution, obj_val = solution_generating_procedure(result_list, 3, 5)

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
                                         affected_movements, TIME_INTERVAL, TIME_WINDOW]
            else:
                print("No solution found for instance", instance)
                df.loc[len(df.index)] = [instance, None, None, None, epochs, None, neighborhood_size, t0, alpha,
                                         neighbor_deviation_scale, affected_movements, TIME_INTERVAL, TIME_WINDOW]

        instance += 1

        try:
            df.to_excel('results/SA/output_40x100_T0xAlpha_continued_1.xlsx', index=False)
        except PermissionError:
            print("Please close the file output.xlsx and try again")

        print("solutions found: ", len(df.index))
