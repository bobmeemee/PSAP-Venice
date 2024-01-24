import pickle

import numpy as np
import pandas as pd
from PSAP.Problem import obj_func, collect_instance_data, decimal_to_time, prepare_movements, SolutionGenerator
from PSAP_TABU import generate_parameters, TabuSearch
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
            generate_parameters([1, 100], [1, 100], [2, 18]))

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
    # unseen intances are from 151 to 200
    instance = 151
    sol_found = 0

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'obj_val',
                               'tabu_list_size', 'number_of_tweaks', 'affected_movements', 'epochs',
                               'time_interval', 'vessel_time_window', 'solution_found', 'predicted_delay'])

    time_window = 60 * 6
    time_interval = 5

    # load the ANN regressor as a .pth file
    net = Net()
    PATH = '../../results/TABU/models/NN_model_150-correctInst.pth'
    net.load_state_dict(torch.load(PATH))

    # load the scaler
    scaler = pickle.load(open('../../results/TABU/models/scaler_150-correctInst.pkl', 'rb'))

    # initialize the solver
    solver = TabuSearch(max_time=60 * 60, time_interval=time_interval, vessel_time_window=time_window,
                        tabu_list_size=10, number_of_tweaks=10, affected_movements=10, epochs=1000)

    # initialize the solution generator
    solutionGenerator = SolutionGenerator(movements=[], l=3, t=5, solver=solver, time_interval=5,
                                          vessel_time_window=60 * 6)

    while instance < 201:
        print("=====================================")
        print("Instance: ", instance)

        # read in the data
        result_list = prepare_movements(instance)
        result_dict = {m: m.optimal_time for m in result_list}


        print("Objective value initial solution: ", obj_func(result_dict))

        instance_data = collect_instance_data(result_list)
        print(instance_data)

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(10):

            # generate good parameters
            epochs = 1000
            treshold = -np.inf
            max_time = 60 * 5
            tabu_list_size, number_of_tweaks, affected_movements, pred = generate_good_params_ANN(net, scaler,
                                                                                                  instance_data,
                                                                                                  treshold,
                                                                                                  max_epochs=30000)

            # pass the parameters to the solver
            solver.set_tabu_list_size(tabu_list_size)
            solver.set_number_of_tweaks(number_of_tweaks)
            solver.set_affected_movements(affected_movements)

            # run the solution generating procedure
            solutionGenerator.set_movements(result_list.copy())

            initial_solution, obj_val, prev_initial_solution = solutionGenerator.generate_solution()


            # save the results
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

        df.to_excel('results/TABU/ML-Results/ignore.xlsx', index=False)
        print("solutions found: ", sol_found, "/", len(df.index))
