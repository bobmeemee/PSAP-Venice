import pandas as pd
import time
import random as rd
import numpy as np
from matplotlib import pyplot as plt

from Problem import Movement, time_to_decimal, decimal_to_time, validate_solution, generate_initial_solution, \
    read_data, obj_func, earliest

TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6
ZAZA = 0


class TabuList:
    def __init__(self, size):
        self.size = size
        self.tabu_list = []

    def add(self, element):
        if len(self.tabu_list) >= self.size:
            self.tabu_list.pop(0)
        self.tabu_list.append(element)

    def contains(self, element):
        return element in self.tabu_list


def generate_neighbor_solution_TABU(initial_solution, affected_movements, deviation_scale, time_interval=5):
    chosen_neighbors = []
    if affected_movements > len(initial_solution):
        affected_movements = len(initial_solution)

    for _ in range(affected_movements):
        # choose a random movement
        movement = rd.choice(list(initial_solution.keys()))
        while movement in chosen_neighbors:
            movement = rd.choice(list(initial_solution.keys()))
        # choose a random time interval
        time_deviation = round(rd.gauss(0, deviation_scale) / time_interval) * time_interval

        # apply the time deviation
        initial_solution[movement] += time_deviation / 60
        chosen_neighbors.append(movement)

    return initial_solution, chosen_neighbors


# ============TABU SEARCH================
def solve_with_precedence_constraints_TABU(movements: list, precedence: dict, max_time: int,
                                           tabu_list_size=3, number_of_tweaks=10,
                                           affected_movements=4,
                                           time_interval=5, vessel_time_window=60 * 6):
    # start the timer
    start_time = time.time()

    # the initial solution is a dictionary with movements as keys and scheduled times as values
    initial_solution = {m: m.optimal_time for m in movements}
    initial_obj_val = obj_func(initial_solution, precedence)

    best_solution = initial_solution.copy()
    best_obj_val = initial_obj_val

    # initialize the tabu list
    tabu_list = TabuList(tabu_list_size)

    if validate_solution(initial_solution, vessel_time_window):
        return initial_solution, initial_obj_val

    # while the time limit is not reached
    while time.time() - start_time < max_time and not validate_solution(initial_solution, vessel_time_window):
        # generate a new solution
        r = initial_solution.copy()
        r, chosen_movements_r = generate_neighbor_solution_TABU(r, affected_movements=affected_movements,
                                                                deviation_scale=40,
                                                                time_interval=time_interval)
        while chosen_movements_r in tabu_list.tabu_list:
            r = initial_solution.copy()
            r, chosen_movements_r = generate_neighbor_solution_TABU(r, affected_movements=affected_movements,
                                                                    deviation_scale=40,
                                                                    time_interval=time_interval)

        r_obj_val = obj_func(r, precedence)
        chosen_movements_r = sorted(chosen_movements_r, key=lambda x: x.optimal_time)
        for idx in range(number_of_tweaks):
            w = initial_solution.copy()
            w, chosen_movements_w = generate_neighbor_solution_TABU(w, affected_movements=affected_movements,
                                                                    deviation_scale=40,
                                                                    time_interval=time_interval)
            while chosen_movements_w in tabu_list.tabu_list:
                w = initial_solution.copy()
                w, chosen_movements_w = generate_neighbor_solution_TABU(w, affected_movements=affected_movements,
                                                                        deviation_scale=40,
                                                                        time_interval=time_interval)
            w_obj_val = obj_func(w, precedence)
            if w_obj_val < r_obj_val:
                r = w.copy()
                r_obj_val = w_obj_val
                chosen_movements_r = chosen_movements_w.copy()

        initial_solution = r.copy()
        initial_obj_val = r_obj_val
        tabu_list.add(chosen_movements_r)

        if initial_obj_val < best_obj_val:
            best_solution = initial_solution.copy()
            best_obj_val = initial_obj_val

    if validate_solution(initial_solution, vessel_time_window, print_errors=True):
        return best_solution, initial_obj_val
    else:
        return None, None


def solution_generating_procedure(movements: list, l, t,
                                  tabu_list_size=3, number_of_tweaks=10,
                                  affected_movements=4,
                                  time_interval=5, vessel_time_window=60 * 6):
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
        solution, obj_val = solve_with_precedence_constraints_TABU(problem_subset, precedence, max_time=t,
                                                                   time_interval=time_interval,
                                                                   tabu_list_size=tabu_list_size,
                                                                   number_of_tweaks=number_of_tweaks,
                                                                   affected_movements=affected_movements,

                                                                   vessel_time_window=vessel_time_window)

        # if no solution was found, return None
        if solution is None:
            print("No solution found while using precedence constraints")
            print("reached: ", len(fixed_movements), "/", len(movements))
            try:
                return None, prev_obj_val, prev_solution
            except UnboundLocalError:
                # return infinite
                return None, 10 ** 9, {}

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


def generate_parameters(tabu_list_size_rng, number_of_tweaks_rng, affected_movements_rng):
    tabu_list_size = rd.randint(tabu_list_size_rng[0], tabu_list_size_rng[1])
    number_of_tweaks = rd.randint(number_of_tweaks_rng[0], number_of_tweaks_rng[1])
    affected_movements = rd.randint(affected_movements_rng[0], affected_movements_rng[1])

    return tabu_list_size, number_of_tweaks, affected_movements


if __name__ == '__main__':
    sol_found = False
    instance = 1
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'obj_val',
                               'tabu_list_size', 'number_of_tweaks', 'affected_movements',
                               'time_interval', 'vessel_time_window', 'solution_found'])

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
            # generate the parameters
            tabu_list_size, number_of_tweaks, affected_movements = generate_parameters([2, 6], [3, 10], [2, 6])

            initial_solution, obj_val, prev_initial_solution = \
                solution_generating_procedure(result_list, 3, 5,
                                              tabu_list_size=tabu_list_size,
                                              number_of_tweaks=number_of_tweaks,
                                              affected_movements=affected_movements,
                                              time_interval=time_interval,
                                              vessel_time_window=time_window)

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
                                         obj_val, tabu_list_size, number_of_tweaks, affected_movements,
                                         time_interval, time_window, 1]

                sol_found += 1
            else:
                print("No solution found for instance", instance)
                for m, t in prev_initial_solution.items():
                    m.set_scheduled_time(t)
                obj_val = obj_func(prev_initial_solution)
                df.loc[len(df.index)] = [instance, len(prev_initial_solution),
                                         np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         obj_val, tabu_list_size, number_of_tweaks, affected_movements,
                                         time_interval, time_window, 0]
        instance += 1

        try:
            df.to_excel('results/TABU/output_100e.xlsx', index=False)
        except PermissionError:
            print("Please close the file output.xlsx and try again")

        print("solutions found: ", sol_found, "/", len(df.index))
