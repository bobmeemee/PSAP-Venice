import pandas as pd
import time
import random as rd
import numpy as np
from matplotlib import pyplot as plt

from Problem import Movement, time_to_decimal, decimal_to_time, validate_solution, generate_initial_solution, \
    read_data, generate_neighbor_solution, obj_func, validate_precedence_constraints, earliest, collect_instance_data

TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6


# ============ITERATED LOCAL SEARCH================
def solve_with_precedence_constraints_ILS(movements: list, precedence: dict, max_time: int, epochs,
                                          iteration_time_limit=4,
                                          time_interval=5, vessel_time_window=60 * 6):
    # start the timer
    start_time = time.time()

    # the initial solution is a dictionary with movements as keys and scheduled times as values
    initial_solution = {m: m.optimal_time for m in movements}
    initial_obj_val = obj_func(initial_solution, precedence)
    homebase = initial_solution
    best_solution = initial_solution

    obj_val = []

    # while the time limit is not reached
    while time.time() - start_time < max_time and not validate_solution(initial_solution, vessel_time_window):

        # start the timer for the iteration
        iteration_start_time = time.time()
        while time.time() - iteration_start_time < iteration_time_limit:
            # generate a new solution that can escape the local optimum
            new_solution = initial_solution.copy()
            new_solution = generate_neighbor_solution(new_solution, affected_movements=6,
                                                      deviation_scale=40,
                                                      time_interval=time_interval)

            if obj_func(new_solution, precedence) < obj_func(initial_solution, precedence):
                initial_solution = new_solution.copy()
                initial_obj_val = obj_func(initial_solution, precedence)
                obj_val.append(initial_obj_val)

            # ideal solution found
            if validate_solution(initial_solution, vessel_time_window):
                return initial_solution, initial_obj_val

        if obj_func(initial_solution, precedence) < obj_func(best_solution, precedence):
            best_solution = initial_solution.copy()
            homebase = initial_solution.copy()
            # perturb the solution
            homebase = generate_neighbor_solution(homebase, affected_movements=4,
                                                  deviation_scale=40,
                                                  time_interval=time_interval)
        else:
            initial_solution = homebase.copy()
            initial_obj_val = obj_func(initial_solution, precedence)

        # plot
        plt.plot(obj_val)
        plt.show()

        # if abs(new_obj_val - initial_obj_val) < 0.000001:
        #     stopping_condition_counter += 1
        #     if stopping_condition_counter == 10 * neighborhood_size:
        #         initial_solution = {m: m.optimal_time for m in movements}
        #         initial_obj_val = obj_func(initial_solution)
        #         e = 0
        #         attempt += 1
    #

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
        solution, obj_val = solve_with_precedence_constraints_ILS(problem_subset, precedence, max_time=t,
                                                                  epochs=epochs,
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


def generate_parameters(epochs_rng):
    epochs = rd.randint(epochs_rng[0], epochs_rng[1])
    return epochs


if __name__ == '__main__':
    sol_found = False
    instance = 1
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'epochs', 'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window', 'valid_solution'])

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
        for _ in range(3):
            epochs = generate_parameters(epochs_rng=[100, 100])

            initial_solution, obj_val, prev_initial_solution = solution_generating_procedure(result_list, 3, 5,
                                                                                             epochs=epochs,
                                                                                             time_interval=5,
                                                                                             vessel_time_window=TIME_WINDOW)

            if initial_solution is not None:
                # set the movement scheduled to the result of the solution generating procedure
                for m, t in initial_solution.items():
                    m.set_scheduled_time(t)
                print("Solution", _, " found for instance", instance, "(", len(initial_solution), ")")
                obj_val = obj_func(initial_solution)
                print("Objective value: ", obj_val)
                # df.loc[len(df.index)] = [instance, len(initial_solution),
                #                          np.median([abs(m.get_delay()) for m in initial_solution.keys()]),
                #                          np.mean([abs(m.get_delay()) for m in initial_solution.keys()]),
                #                          epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                #                          affected_movements, TIME_INTERVAL, TIME_WINDOW, 1]
            else:
                print("No solution found for instance", instance)
                for m, t in prev_initial_solution.items():
                    m.set_scheduled_time(t)
                obj_val = obj_func(prev_initial_solution)

                # df.loc[len(df.index)] = [instance, len(prev_initial_solution),
                #                          np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                #                          np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                #                          epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                #                          affected_movements, TIME_INTERVAL, TIME_WINDOW, 0]

        instance += 1

        # try:
        #     df.to_excel('results/ILS/output_100e.xlsx', index=False)
        # except PermissionError:
        #     print("Please close the file output.xlsx and try again")

        print("solutions found: ", len(df.index))
