import pandas as pd
import time
import random as rd
import numpy as np

from PSAP.Problem import (validate_solution, obj_func, Solver,
                          SolutionGenerator, prepare_movements, generate_neighbor_solution)


# ============ITERATED LOCAL SEARCH================

class IteratedLocalSearch(Solver):
    def __init__(self, max_time, time_interval, vessel_time_window, iteration_time_limit=4,
                 neighbor_affected_movements=6, neighbor_deviation_scale=40,
                 homebase_affected_movements=4, homebase_deviation_scale=40):
        super().__init__(max_time, time_interval, vessel_time_window)
        self.iteration_time_limit = iteration_time_limit
        self.neighbor_affected_movements = neighbor_affected_movements
        self.neighbor_deviation_scale = neighbor_deviation_scale
        self.homebase_affected_movements = homebase_affected_movements
        self.homebase_deviation_scale = homebase_deviation_scale

    def generate_parameters(self, neighbor_deviation_scale_rng, neighbor_affected_movements_rng,
                            homebase_deviation_scale_rng,
                            homebase_affected_movements_rng):
        # generate the parameters
        self.neighbor_deviation_scale = rd.randint(neighbor_deviation_scale_rng[0], neighbor_deviation_scale_rng[1])
        self.neighbor_affected_movements = rd.randint(neighbor_affected_movements_rng[0],
                                                      neighbor_affected_movements_rng[1])
        self.homebase_deviation_scale = rd.randint(homebase_deviation_scale_rng[0], homebase_deviation_scale_rng[1])
        self.homebase_affected_movements = rd.randint(homebase_affected_movements_rng[0],
                                                      homebase_affected_movements_rng[1])

        # make sure that deviation is in intervals of 5
        self.neighbor_deviation_scale = round(self.neighbor_deviation_scale / 5) * 5
        self.homebase_deviation_scale = round(self.homebase_deviation_scale / 5) * 5

        return (self.neighbor_deviation_scale, self.neighbor_affected_movements, self.homebase_deviation_scale,
                self.homebase_affected_movements)



    def solve(self, movements, precedence=None):
        # start the timer
        start_time = time.time()

        # the initial solution is a dictionary with movements as keys and scheduled times as values
        initial_solution = {m: m.optimal_time for m in movements}
        initial_obj_val = obj_func(initial_solution, precedence)
        homebase = initial_solution
        best_solution = initial_solution

        obj_val = []

        # while the time limit is not reached
        while time.time() - start_time < self.max_time and not validate_solution(initial_solution,
                                                                                 self.vessel_time_window):

            # start the timer for the iteration
            iteration_start_time = time.time()
            while time.time() - iteration_start_time < self.iteration_time_limit:
                # generate a new solution that can escape the local optimum
                new_solution = initial_solution.copy()
                new_solution = generate_neighbor_solution(new_solution,
                                                          affected_movements=self.neighbor_affected_movements,
                                                          deviation_scale=self.neighbor_deviation_scale,
                                                          time_interval=self.time_interval)

                if obj_func(new_solution, precedence) < obj_func(initial_solution, precedence):
                    initial_solution = new_solution.copy()
                    initial_obj_val = obj_func(initial_solution, precedence)
                    obj_val.append(initial_obj_val)

                # ideal solution found
                if validate_solution(initial_solution, self.vessel_time_window):
                    return initial_solution, initial_obj_val

            if obj_func(initial_solution, precedence) < obj_func(best_solution, precedence):
                best_solution = initial_solution.copy()
                homebase = initial_solution.copy()
                # perturb the solution
                homebase = generate_neighbor_solution(homebase, affected_movements=self.homebase_affected_movements,
                                                      deviation_scale=self.homebase_deviation_scale,
                                                      time_interval=time_interval)
            else:
                initial_solution = homebase.copy()
                initial_obj_val = obj_func(initial_solution, precedence)

        if validate_solution(best_solution, self.vessel_time_window, print_errors=True):
            return best_solution, obj_func(best_solution)
        else:
            return None, None


if __name__ == '__main__':
    sol_found = False
    instance = 1
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'obj_val',
                               'iteration_time_limit', 'neighbor_deviation_scale',
                               'neighbor_affected_movements', 'homebase_deviation_scale', 'homebase_affected_movements',
                               'time_interval', 'vessel_time_window', 'solution_found'])

    df_instance = pd.DataFrame(columns=['instance', 'number_of_movements', 'number_of_vessels', 'average_headway',
                                        'std_dev_headway', 'spread', 'average_time_between_movements',
                                        'average_travel_time'])

    time_window = 60 * 6
    time_interval = 10

    solver = IteratedLocalSearch(max_time=5, time_interval=time_interval, vessel_time_window=time_window,
                                 iteration_time_limit=4, neighbor_affected_movements=6, neighbor_deviation_scale=40,
                                 homebase_affected_movements=4, homebase_deviation_scale=40)

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
            neighbor_deviation_scale, neighbor_affected_movements, homebase_deviation_scale, homebase_affected_movements \
                = solver.generate_parameters(neighbor_deviation_scale_rng=[30, 50],
                                             neighbor_affected_movements_rng=[5, 7],
                                             homebase_deviation_scale_rng=[20, 45],
                                             homebase_affected_movements_rng=[3, 5])

            print([neighbor_deviation_scale, neighbor_affected_movements, homebase_deviation_scale,
                   homebase_affected_movements])

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
                                         obj_val, 4, neighbor_deviation_scale, neighbor_affected_movements,
                                         homebase_deviation_scale, homebase_affected_movements, time_interval,
                                         time_window, 1]
                sol_found += 1
            else:
                print("No solution found for instance", instance)
                for m, t in prev_initial_solution.items():
                    m.set_scheduled_time(t)
                obj_val = obj_func(prev_initial_solution)
                df.loc[len(df.index)] = [instance, len(prev_initial_solution),
                                         np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         obj_val, 4, neighbor_deviation_scale, neighbor_affected_movements,
                                         homebase_deviation_scale, homebase_affected_movements, time_interval,
                                         time_window, 0]

        instance += 1

        try:
            df.to_excel('../../results/ILS/ignore.xlsx', index=False)
        except PermissionError:
            print("Please close the file output.xlsx and try again")

        print("solutions found: ", sol_found, "/", len(df.index))
