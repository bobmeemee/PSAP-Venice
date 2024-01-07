import time
import pickle

import numpy as np
import pandas as pd
from Problem import read_data, obj_func, collect_instance_data, decimal_to_time
from PSAP_TABU import solution_generating_procedure, generate_parameters, TIME_INTERVAL, TIME_WINDOW, \
    generate_initial_solution
import torch
import torch.nn as nn
import torch.nn.functional as F


# define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def generate_good_params(regressor, instance_data, treshold, max_time):
    pred = np.inf
    t = time.time()
    elapsed_time = 0
    best_alpha = 0
    best_t0 = 0
    best = np.inf
    while pred > treshold and elapsed_time < max_time:
        # generate parameters
        tabu_list_size, number_of_tweaks, affected_movements = (
            generate_parameters([1, 5], [6, 10], [2, 4]))

        # concatenate the parameters to the instance data
        instance_data['tabu_list_size'] = tabu_list_size
        instance_data['number_of_tweaks'] = number_of_tweaks
        instance_data['affected_movements'] = affected_movements

        # dict to list in order to be able to predict
        feature_cols = ['tabu_list_size', 'number_of_tweaks', 'affected_movements',
                        'number_of_movements', 'number_of_vessels', 'average_headway', 'std_dev_headway', 'spread',
                        'average_time_between_movements', 'average_travel_time']
        features = []
        for col in feature_cols:
            features.append(instance_data[col])

        predictions = regressor.predict([features])
        pred = predictions[0]
        print("Predicted delay: ", decimal_to_time(pred))
        elapsed_time = time.time() - t
        if pred < best:
            best = pred
            best_t0 = t0
            best_alpha = alpha

    if pred > treshold:
        return None, None, None

    return best_alpha, best_t0, best


def generate_good_params_ANN(net, scaler, instance_data, treshold, max_epochs=30000):
    pred = np.inf
    best = np.inf
    # t = time.time()
    # elapsed_time = 0
    best_tabu_list_size = 0
    best_number_of_tweaks = 0
    best_affected_movements = 0
    e = 0
    print("Looking for good parameters...")
    while best > treshold and e < max_epochs:
        # generate parameters
        tabu_list_size, number_of_tweaks, affected_movements = (
            generate_parameters([1, 5], [6, 10], [2, 4]))

        # concatenate the parameters to the instance data
        instance_data['tabu_list_size'] = tabu_list_size
        instance_data['number_of_tweaks'] = number_of_tweaks
        instance_data['affected_movements'] = affected_movements

        # dict to list in order to be able to predict
        feature_cols = ['tabu_list_size', 'number_of_tweaks', 'affected_movements',
                        'number_of_movements', 'number_of_vessels', 'average_headway', 'std_dev_headway', 'spread',
                        'average_time_between_movements', 'average_travel_time']
        features = []
        for col in feature_cols:
            features.append(instance_data[col])

        features = scaler.transform([features])

        predictions = net(torch.tensor(features, dtype=torch.float32))
        pred = predictions.item()
        if pred < best:
            best = pred
            best_tabu_list_size = tabu_list_size
            best_number_of_tweaks = number_of_tweaks
            best_affected_movements = affected_movements

        e += 1

        # elapsed_time = time.time() - t

    if best < treshold:
        print("Good parameters found")

    print("epochs reached: ", e)

    return best_tabu_list_size, best_number_of_tweaks, best_affected_movements, best


if __name__ == '__main__':
    sol_found = False
    # unseen intances are from 151 to 200
    instance = 51
    sol_found = 0

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'obj_val',
                               'tabu_list_size', 'number_of_tweaks', 'affected_movements', 'epochs',
                               'time_interval', 'vessel_time_window', 'solution_found', 'predicted_delay'])

    time_window = 60 * 6
    time_interval = 5

    # load the ANN regressor as a .pth file
    net = Net()
    PATH = 'results/TABU/models/NN_model_150.pth'
    net.load_state_dict(torch.load(PATH))

    # load the scaler
    scaler = pickle.load(open('results/TABU/models/scaler_150.pkl', 'rb'))

    while instance < 101:
        print("=====================================")
        print("Instance: ", instance + 100)

        # read in the data
        df_movimenti, df_precedenze, df_tempi = read_data(instance)
        # generate the initial solution
        initial_solution = generate_initial_solution(df=df_movimenti, df_headway=df_precedenze, deviation_scale=1,
                                                     time_interval=TIME_INTERVAL, df_tempi=df_tempi)

        movements = list(initial_solution.keys())
        sorted_movements = sorted(movements, key=lambda x: x.optimal_time)
        # this are the instances above 100
        result_list = [elem for index, elem in enumerate(sorted_movements, 1) if index % 2 == 0]

        print("Objective value initial solution: ", obj_func(initial_solution))

        instance_data = collect_instance_data(result_list)
        print(instance_data)

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(10):

            # generate good parameters
            epochs = 1000
            treshold = 0.8
            max_time = 60 * 5
            tabu_list_size, number_of_tweaks, affected_movements, pred = generate_good_params_ANN(net, scaler,
                                                                                                  instance_data,
                                                                                                  treshold,
                                                                                                  max_epochs=30000)

            initial_solution, obj_val, prev_initial_solution = \
                solution_generating_procedure(result_list, 3, 5,
                                              tabu_list_size=tabu_list_size,
                                              number_of_tweaks=number_of_tweaks,
                                              affected_movements=affected_movements,
                                              epochs=epochs,
                                              time_interval=time_interval,
                                              vessel_time_window=time_window)

            if initial_solution is not None:
                # set the movement scheduled to the result of the solution generating procedure
                for m, t in initial_solution.items():
                    m.set_scheduled_time(t)
                print("Solution", _, " found for instance", instance + 100, "(", len(initial_solution), ")")
                obj_val = obj_func(initial_solution)
                print("Objective value: ", obj_val)
                avg_delay = np.mean([abs(m.get_delay()) for m in initial_solution.keys()])
                med_delay = np.median([abs(m.get_delay()) for m in initial_solution.keys()])
                print("Average delay: ", decimal_to_time(avg_delay))
                print("Predicted delay: ", decimal_to_time(pred))
                print("Median delay: ", decimal_to_time(med_delay))
                df.loc[len(df.index)] = [instance, len(initial_solution),
                                         np.median([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         obj_val, tabu_list_size, number_of_tweaks, affected_movements, epochs,
                                         time_interval, time_window, 1, pred]
                sol_found += 1
            else:
                for m, t in prev_initial_solution.items():
                    m.set_scheduled_time(t)
                obj_val = obj_func(prev_initial_solution)

                df.loc[len(df.index)] = [instance, len(prev_initial_solution),
                                         np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         obj_val, tabu_list_size, number_of_tweaks, affected_movements, epochs,
                                         time_interval, time_window, 0, pred]
                print("No solution found for instance", instance + 100)

        instance += 1

        df.to_excel('results/TABU/ML-Results/ML151-200x10.xlsx', index=False)
        print("solutions found: ", sol_found, "/", len(df.index))
