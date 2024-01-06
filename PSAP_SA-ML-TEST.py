import time
import pickle

import numpy as np
import pandas as pd
from Problem import read_data, obj_func, collect_instance_data, decimal_to_time
from PSAP_SA import solution_generating_procedure, generate_parameters, TIME_INTERVAL, TIME_WINDOW, \
    generate_initial_solution

import torch
import torch.nn as nn
import torch.nn.functional as F


# define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def generate_good_params(regressor, instance_data, treshold, max_time):
    pred = 1000
    t = time.time()
    elapsed_time = 0
    best_alpha = 0
    best_t0 = 0
    best = np.inf
    while pred > treshold and elapsed_time < max_time:
        # generate parameters
        epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements = generate_parameters(
            epochs_rng=[100, 100],
            neighborhood_size_rng=[4, 4],
            t0_rng=[30, 500],
            alpha_rng=[0.4, 0.99],
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
        if pred < best:
            best = pred
            best_t0 = t0
            best_alpha = alpha

    if pred > treshold:
        return None, None, None

    return best_alpha, best_t0, best


def generate_good_params_ANN(net, scaler, instance_data, treshold, max_time, max_epochs=200):
    pred = np.inf
    best = np.inf
    # t = time.time()
    # elapsed_time = 0
    best_alpha = 0
    best_t0 = 0
    e = 0
    print("Looking for good parameters...")
    while best > treshold and e < max_epochs:
        # generate parameters
        epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements = generate_parameters(
            epochs_rng=[100, 100],
            neighborhood_size_rng=[4, 4],
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

        features = scaler.transform([features])

        predictions = net(torch.tensor(features, dtype=torch.float32))
        pred = predictions.item()
        if pred < best:
            best = pred
            best_t0 = t0
            best_alpha = alpha
        e += 1

        # elapsed_time = time.time() - t

    if best < treshold:
        print("Good parameters found")

    print("epochs reached: ", e)

    return best_alpha, best_t0, best


if __name__ == '__main__':
    sol_found = False
    # unseen intances are from 151 to 200
    instance = 51
    sol_found = 0

    df = pd.DataFrame(columns=['instance', 'number_of_movements_reached', 'median_delay', 'average_delay', 'epochs',
                               'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window', 'valid_solution', 'predicted_delay'])

    time_window = 60 * 6
    time_interval = 5

    # load the ANN regressor as a .pth file
    net = Net()
    PATH = 'results/SA/models/NN_model_150e.pth'
    net.load_state_dict(torch.load(PATH))

    # load the scaler
    scaler = pickle.load(open('results/SA/models/scaler_150e.pkl', 'rb'))

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
        for _ in range(1):
            epochs, neighborhood_size, t0, alpha, neighbor_deviation_scale, affected_movements = generate_parameters(
                epochs_rng=[100, 100],
                neighborhood_size_rng=[6, 6],
                t0_rng=[40, 500],
                alpha_rng=[0.6, 0.99],
                neighbor_deviation_scale_rng=[40, 40],
                affected_movements_rng=[4, 4])

            alpha, t0, pred = generate_good_params_ANN(net, scaler, instance_data, .6, 10, max_epochs=9000)

            initial_solution, obj_val, prev_initial_solution = solution_generating_procedure(result_list, 3, 5,
                                                                                             epochs=epochs,
                                                                                             neighborhood_size=neighborhood_size,
                                                                                             t0=t0, alpha=alpha,
                                                                                             neighbor_deviation_scale=neighbor_deviation_scale,
                                                                                             affected_movements=affected_movements,
                                                                                             time_interval=5,
                                                                                             vessel_time_window=TIME_WINDOW)

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
                df.loc[len(df.index)] = [instance + 100, len(initial_solution),
                                         np.median([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in initial_solution.keys()]),
                                         epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                                         affected_movements, TIME_INTERVAL, TIME_WINDOW, 1, pred]
                sol_found += 1
            else:
                df.loc[len(df.index)] = [instance + 100, len(prev_initial_solution),
                                         np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                                         affected_movements, TIME_INTERVAL, TIME_WINDOW, 0, pred]
                print("No solution found for instance", instance +100)

        instance += 1

    df.to_excel('results/SA/ML-Results/out_151-200ex10.xlsx', index=False)
    print("solutions found: ", sol_found, "/", len(df.index))
