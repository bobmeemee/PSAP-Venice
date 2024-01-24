import pickle

import numpy as np
import pandas as pd
from PSAP.Problem import obj_func, collect_instance_data, decimal_to_time, SolutionGenerator, \
    prepare_movements
from PSAP.PSAP_SA import generate_parameters, SimulatedAnnealing

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


def generate_good_params_ANN(net, scaler, instance_data, treshold, max_time, max_epochs=200):
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
            t0_rng=[1, 10000],
            alpha_rng=[0.01, 0.99],
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
    # unseen intances are from 151 to 200
    instance = 151
    sol_found = 0

    df = pd.DataFrame(columns=['instance', 'number_of_movements_reached', 'median_delay', 'average_delay', 'epochs',
                               'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window', 'valid_solution', 'predicted_delay'])

    time_window = 60 * 6
    time_interval = 5

    # load the ANN regressor as a .pth file
    net = Net()
    PATH = '../../results/SA/models/NN_model_150e.pth'
    net.load_state_dict(torch.load(PATH))

    # load the scaler
    scaler = pickle.load(open('../../results/SA/models/scaler_150e.pkl', 'rb'))

    # initialize the solver
    solver = SimulatedAnnealing(max_time=60, epochs=100,
                                time_interval=TIME_INTERVAL, vessel_time_window=TIME_WINDOW,
                                neighborhood_size=4, t0=1000, alpha=0.5, neighbor_deviation_scale=40,
                                affected_movements=4)

    # initialize the solution generator
    solutionGenerator = SolutionGenerator(movements=[], l=3, t=5, solver=solver, time_interval=5,
                                          vessel_time_window=60 * 6)

    while instance < 201:
        print("=====================================")
        print("Instance: ", instance)

        # read in the data
        result_list = prepare_movements(instance)
        result_dict = {m: m.optimal_time for m in result_list}
        lengthMaxMov = len(result_list)

        print("Objective value initial solution: ", obj_func(result_dict))

        instance_data = collect_instance_data(result_list)
        print(instance_data)

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(1):

            epochs = 100
            neighborhood_size = 4
            neighbor_deviation_scale = 40
            affected_movements = 4

            alpha, t0, pred = generate_good_params_ANN(net, scaler, instance_data, .6, 10, max_epochs=30000)

            solver.set_t0(t0)
            solver.set_alpha(alpha)

            solutionGenerator.set_movements(result_list.copy())
            initial_solution, obj_val, prev_initial_solution = solutionGenerator.generate_solution()

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
                for m, t in prev_initial_solution.items():
                    m.set_scheduled_time(t)
                obj_val = obj_func(prev_initial_solution)

                df.loc[len(df.index)] = [instance + 100, len(prev_initial_solution),
                                         np.median([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         np.mean([abs(m.get_delay()) for m in prev_initial_solution.keys()]),
                                         epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                                         affected_movements, TIME_INTERVAL, TIME_WINDOW, 0, pred]
                print("No solution found for instance", instance + 100)

        instance += 1

        df.to_excel('results/SA/ML-Results/out_151-200ex10-Bavo.xlsx', index=False)
        print("solutions found: ", sol_found, "/", len(df.index))
