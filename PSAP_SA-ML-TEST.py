import time
import pickle

import numpy as np
import pandas as pd
from Problem import read_data, obj_func, collect_instance_data, decimal_to_time
from PSAP_SA import solution_generating_procedure, generate_parameters, TIME_INTERVAL, TIME_WINDOW, \
    generate_initial_solution



def generate_good_params(regressor, instance_data, treshold, max_time):
    pred = 1000
    t = time.time()
    elapsed_time = 0
    alpha = 0
    t0 = 0
    while pred > treshold and elapsed_time < max_time:
        # generate parameters
        epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements = generate_parameters(
            epochs_rng=[200, 200],
            neighborhood_size_rng=[6, 6],
            t0_rng=[40, 500],
            alpha_rng=[0.6, 0.99],
            neighbor_deviation_scale_rng=[40, 40],
            affected_movements_rng=[4, 4])

        # concatenate the parameters to the instance data
        instance_data['t0'] = t0
        instance_data['alpha'] = alpha

        # dict to list in order to be able to predict
        feature_cols = ['t0', 'alpha',
                        'number_of_movements', 'number_of_vessels', 'average_headway', 'std_dev_headway', 'spread',
                        'average_time_between_movements', 'average_travel_time']
        features = []
        for col in feature_cols:
            features.append(instance_data[col])

        predictions = regressor.predict([features])
        pred = predictions[0]
        print("Predicted delay: ", decimal_to_time(pred))
        elapsed_time = time.time() - t

    if pred > treshold:
        return None, None, None

    return alpha, t0, pred


if __name__ == '__main__':
    sol_found = False
    instance = 1
    sol_found = 0

    data = pd.DataFrame(columns=['instance', 'AI_solved', 'obj_val', 'avg_delay', 'med_delay', 't0', 'alpha'])

    time_window = 60 * 6
    time_interval = 5

    # collect the regressor from the file
    with open('results/SA/models/RandomForestRegressor.sav', 'rb') as f:
        regressor = pickle.load(f)

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
        result_list = [elem for index, elem in enumerate(sorted_movements, 1) if index % 2 == 0]

        print("Objective value initial solution: ", obj_func(initial_solution))

        instance_data = collect_instance_data(result_list)
        print(instance_data)

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(1):
            epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements = generate_parameters(
                epochs_rng=[200, 200],
                neighborhood_size_rng=[6, 6],
                t0_rng=[40, 500],
                alpha_rng=[0.6, 0.99],
                neighbor_deviation_scale_rng=[40, 40],
                affected_movements_rng=[4, 4])

            alpha, t0, pred = generate_good_params(regressor, instance_data, 0.9, 10)

            if alpha is None:
                print("No alpha/t0 found for instance", instance)
                instance += 1
                continue

            initial_solution, obj_val = solution_generating_procedure(result_list, 3, 5, epochs=epochs,
                                                                      neighborhood_size=neighborhood_size,
                                                                      t0=t0, alpha=alpha,
                                                                      neighbor_deviation_scale=neighbor_deviation_scale,
                                                                      affected_movements=affected_movements,
                                                                      time_interval=5, vessel_time_window=TIME_WINDOW)

            if initial_solution is not None:
                # set the movement scheduled to the result of the solution generating procedure
                for m, t in initial_solution.items():
                    m.set_scheduled_time(t)
                print("Solution", _, " found for instance", instance, "(", len(initial_solution), ")")
                obj_val = obj_func(initial_solution)
                print("Objective value: ", obj_val)
                avg_delay = np.mean([abs(m.get_delay()) for m in initial_solution.keys()])
                med_delay = np.median([abs(m.get_delay()) for m in initial_solution.keys()])
                print("Average delay: ", decimal_to_time(avg_delay))
                print("Predicted delay: ", decimal_to_time(pred))
                print("Median delay: ", decimal_to_time(med_delay))
                data.loc[len(data.index)] = [instance, 1, obj_val, avg_delay, med_delay, t0, alpha]
                sol_found += 1
            else:
                data.loc[len(data.index)] = [instance, 0, None, None, None, t0, alpha]
                print("No solution found for instance", instance)

        instance += 1

        print("solutions found: ",sol_found, "/", len(data.index))
