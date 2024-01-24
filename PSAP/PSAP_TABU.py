import pandas as pd
import random as rd
import numpy as np
import math as math
import os

from PSAP.Problem import validate_solution, obj_func, Solver, SolutionGenerator, prepare_movements


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


class TabuSearch(Solver):
    def __init__(self, max_time, tabu_list_size, number_of_tweaks,
                 affected_movements, epochs, time_interval=5*60, vessel_time_window=5):
        super().__init__(max_time, time_interval, vessel_time_window)
        self.tabu_list_size = tabu_list_size
        self.number_of_tweaks = number_of_tweaks
        self.affected_movements = affected_movements
        self.epochs = epochs

    @staticmethod
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

    def generate_parameters(self, tabu_list_size_rng, number_of_tweaks_rng, affected_movements_rng):
        self.tabu_list_size = rd.randint(tabu_list_size_rng[0], tabu_list_size_rng[1])
        self.number_of_tweaks = rd.randint(number_of_tweaks_rng[0], number_of_tweaks_rng[1])
        self.affected_movements = rd.randint(affected_movements_rng[0], affected_movements_rng[1])

        return self.tabu_list_size, self.number_of_tweaks, self.affected_movements

    def solve(self, movements, precedence=None):
        e = 0
        # the initial solution is a dictionary with movements as keys and scheduled times as values
        initial_solution = {m: m.optimal_time for m in movements}
        initial_obj_val = obj_func(initial_solution, precedence)

        best_solution = initial_solution.copy()
        best_obj_val = initial_obj_val

        # initialize the tabu list
        tabu_list = TabuList(self.tabu_list_size)

        if validate_solution(initial_solution, self.vessel_time_window):
            return initial_solution, initial_obj_val

        # bugfix, in this case the tabu list can contain all the possible movements and the algorithm will get stuck
        possible_combinations = math.comb(len(movements), self.affected_movements)
        if possible_combinations <= self.tabu_list_size:
            self.affected_movements += 1

        # while the limit is not reached
        while e < self.epochs and not validate_solution(initial_solution, self.vessel_time_window):
            # generate a new solution
            r = initial_solution.copy()
            r, chosen_movements_r = self.generate_neighbor_solution_TABU(initial_solution=r,
                                                                         affected_movements=self.affected_movements,
                                                                         deviation_scale=40,
                                                                         time_interval=self.time_interval)
            cnt = 0
            while chosen_movements_r in tabu_list.tabu_list:
                r = initial_solution.copy()
                r, chosen_movements_r = self.generate_neighbor_solution_TABU(initial_solution=r,
                                                                             affected_movements=self.affected_movements,
                                                                             deviation_scale=40,
                                                                             time_interval=self.time_interval)

            r_obj_val = obj_func(r, precedence)
            chosen_movements_r = sorted(chosen_movements_r, key=lambda x: x.optimal_time)
            for idx in range(self.number_of_tweaks):
                w = initial_solution.copy()
                w, chosen_movements_w = self.generate_neighbor_solution_TABU(initial_solution=w,
                                                                             affected_movements=self.affected_movements,
                                                                             deviation_scale=40,
                                                                             time_interval=self.time_interval)
                while chosen_movements_w in tabu_list.tabu_list:
                    w = initial_solution.copy()
                    w, chosen_movements_w = self.generate_neighbor_solution_TABU(initial_solution=w,
                                                                                 affected_movements=self.affected_movements,
                                                                                 deviation_scale=40,
                                                                                 time_interval=self.time_interval)
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

            e += 1

        if validate_solution(initial_solution, self.vessel_time_window, print_errors=True):
            return best_solution, initial_obj_val
        else:
            return None, None

    def set_tabu_list_size(self, tabu_list_size):
        self.tabu_list_size = tabu_list_size

    def set_number_of_tweaks(self, number_of_tweaks):
        self.number_of_tweaks = number_of_tweaks

    def set_affected_movements(self, affected_movements):
        self.affected_movements = affected_movements

    def __str__(self):
        out = "Tabu Search\n" + " Tabu list size: " + str(self.tabu_list_size) + "\n" + \
              " Number of tweaks: " + str(self.number_of_tweaks) + "\n" + \
              " Affected movements: " + str(self.affected_movements) + "\n" + \
              " Epochs: " + str(self.epochs) + "\n"
        return out


def generate_parameters(tabu_list_size_rng, number_of_tweaks_rng, affected_movements_rng):
    tabu_list_size = rd.randint(tabu_list_size_rng[0], tabu_list_size_rng[1])
    number_of_tweaks = rd.randint(number_of_tweaks_rng[0], number_of_tweaks_rng[1])
    affected_movements = rd.randint(affected_movements_rng[0], affected_movements_rng[1])

    return tabu_list_size, number_of_tweaks, affected_movements


if __name__ == '__main__':
    sol_found = False
    instance = 51
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'obj_val',
                               'tabu_list_size', 'number_of_tweaks', 'affected_movements', 'epochs',
                               'time_interval', 'vessel_time_window', 'solution_found'])

    df_instance = pd.DataFrame(columns=['instance', 'number_of_movements', 'number_of_vessels', 'average_headway',
                                        'std_dev_headway', 'spread', 'average_time_between_movements',
                                        'average_travel_time'])

    time_window = 60 * 6
    time_interval = 5
    epochs = 1000

    solver = TabuSearch(max_time=60 * 60, time_interval=time_interval, vessel_time_window=time_window,
                        tabu_list_size=10, number_of_tweaks=10, affected_movements=10, epochs=epochs)

    solutionGenerator = SolutionGenerator(movements=[], l=3, t=5, solver=solver, time_interval=5,
                                          vessel_time_window=60 * 6)

    while instance < 101:
        print("=====================================")
        print("Instance: ", instance)

        # read in the data
        result_list = prepare_movements(instance)
        result_dict = {m: m.optimal_time for m in result_list}
        print("Objective value initial solution: ", obj_func(result_dict))


        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(1):
            # generate the parameters

            tabu_list_size, number_of_tweaks, affected_movements = solver.generate_parameters([1, 100],
                                                                                              [1, 100],
                                                                                              [2, 18])
            solutionGenerator.set_movements(result_list.copy())
            initial_solution, obj_val, prev_initial_solution = solutionGenerator.generate_solution()

            # if a solution is found
            if initial_solution is not None:
                # set the movement scheduled to the result of the solution generating procedure
                for m, t in initial_solution.items():
                    m.set_scheduled_time(t)
                print("Solution", _, " found for instance", instance, "(", len(initial_solution), ")")
                obj_val = obj_func(initial_solution)
                print("Objective value: ", obj_val)
                df.loc[len(df.index)] = [instance + 100, len(initial_solution),
                                         np.median([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         obj_val, tabu_list_size, number_of_tweaks, affected_movements, epochs,
                                         time_interval, time_window, 1]

                sol_found += 1
            else:
                print("No solution found for instance", instance)
                for m, t in prev_initial_solution.items():
                    m.set_scheduled_time(t)
                obj_val = obj_func(prev_initial_solution)
                df.loc[len(df.index)] = [instance + 100, len(prev_initial_solution),
                                         np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         obj_val, tabu_list_size, number_of_tweaks, affected_movements, epochs,
                                         time_interval, time_window, 0]
        instance += 1

        try:
            df.to_excel('/results/TABU/testfile_ignore.xlsx', index=False)
        except PermissionError:
            print("Please close the file output.xlsx and try again")

        print("solutions found: ", sol_found, "/", len(df.index))
