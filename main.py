import pandas as pd
import time
import random as rd
import numpy as np

from Problem import Movement, time_to_decimal, decimal_to_time, validate_solution, generate_initial_solution, \
    read_data, generate_neighbor_solution, obj_func, validate_precedence_constraints, earliest

INSTANCE = 19
TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6

T0 = 1000
ALPHA = 0.95
NEIGHBORHOOD_SIZE = 5  # number of neighbours to consider at each iteration
EPOCHS = 300  # number of iterations


# ============SIMULATED ANNEALING================
# method one: generate a random solution and improve on it by also respecting the precedence constraints
def solve_with_precedence_constraints(movements: list, precedence: dict, max_time: int, epochs, neighborhood_size: int,
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
    while time.time() - start_time < max_time and not validate_solution(initial_solution, vessel_time_window):
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
                if p < np.exp(-(new_obj_val - initial_obj_val) / t0):
                    initial_solution = new_solution.copy()
                    initial_obj_val = new_obj_val

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
        # print("Attempt: ", attempt, "Epoch: ", e, "Time: ", time.time() - start_time, "Objective value: ", initial_obj_val)

    if validate_solution(initial_solution, vessel_time_window, print_errors=True):
        return initial_solution, initial_obj_val
    else:
        return None, None


def solution_generating_procedure(movements: list, l, t):
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
        solution, obj_val = solve_with_precedence_constraints(problem_subset, precedence, max_time=t,
                                                              epochs=EPOCHS, neighborhood_size=NEIGHBORHOOD_SIZE,
                                                              t0=T0, alpha=ALPHA,
                                                              neighbor_deviation_scale=40, affected_movements=3,
                                                              time_interval=5, vessel_time_window=TIME_WINDOW)

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
    sol_found = False
    instance = 1
    valid_solutions = []
    while instance < 10:
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

        initial_solution, obj_val = solution_generating_procedure(result_list, 3, 5)

        if initial_solution is not None:
            print("Solution found for instance", instance, "(", len(initial_solution), ")")
            print("Objective value: ", obj_val)
            valid_solutions.append(initial_solution)

        instance += 1
        print("solutions found: ", len(valid_solutions), "/", instance - 1)


"""
    df_movimenti = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='movimenti')
    df_precedenze = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='Precedenze')
    initial_solution, initial_obj_val, temp, obj_val = SA_solve(epochs=EPOCHS, neighborhood_size=NEIGHBORHOOD_SIZE,
                                                                df_movimenti=df_movimenti, df_precedenze=df_precedenze,
                                                                t0=T0, alpha=ALPHA, time_window=TIME_WINDOW,
                                                                time_interval=TIME_INTERVAL, print_errors=True)

    if initial_solution is not None:
        print('Initial solution: ' + str(initial_solution))
        print('Initial objective value: ' + str(initial_obj_val))
        print('Final solution: ' + str(temp))
        print('Final objective value: ' + str(obj_val))

    elif initial_solution is None:
        print('No solution found')
        # TODO implement a different algorithm to find a feasible solution
"""
