import numpy as np
import pandas as pd
import random as rd

from matplotlib import pyplot as plt
from Problem import Movement, time_to_decimal, decimal_to_time, validate_solution, generate_initial_solution, \
    read_data, generate_neighbor_solution, obj_func, validate_precedence_constraints, earliest

INSTANCE = 11
TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6

T0 = 100
ALPHA = 0.98
NEIGHBORHOOD_SIZE = 10  # number of neighbours to consider at each iteration
EPOCHS = 500  # number of iterations

# read in the data
df_movimenti = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='movimenti')
df_precedenze = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='Precedenze')


# ============================SIMULATED ANNEALING================================ #
def SA_solve(epochs, neighborhood_size, df_movimenti, df_precedenze, t0, alpha, time_window, time_interval=5,
             print_errors=False, initial_deviation_scale=1, neighbor_deviation_scale=40, affected_movements=3):
    # generate initial solution
    initial_solution = generate_initial_solution(df_movimenti, df_precedenze, initial_deviation_scale, time_interval)
    initial_obj_val = obj_func(initial_solution)

    temp = []
    obj_val = []

    for idx in range(epochs):
        for jdx in range(neighborhood_size):
            new_solution = initial_solution.copy()
            new_solution = generate_neighbor_solution(new_solution, affected_movements=affected_movements,
                                                      deviation_scale=neighbor_deviation_scale,
                                                      time_interval=time_interval)

            # calculate the objective function of the new solution
            new_obj_val = obj_func(new_solution)

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

        temp.append(t0)
        obj_val.append(initial_obj_val)
        t0 = alpha * t0

    if validate_solution(initial_solution, time_window=time_window, print_errors=print_errors):
        return initial_solution, initial_obj_val, temp, obj_val
    else:
        return None, None, None, None


# generate initial solution
initial_solution = generate_initial_solution(df_movimenti, df_precedenze, 1)
initial_obj_val = obj_func(initial_solution)

print('Initial solution is valid:', validate_solution(initial_solution, TIME_WINDOW))

temp = []
obj_val = []

for idx in range(EPOCHS):
    for jdx in range(NEIGHBORHOOD_SIZE):
        new_solution = initial_solution.copy()
        new_solution = generate_neighbor_solution(new_solution, affected_movements=3, deviation_scale=10)

        # calculate the objective function of the new solution
        new_obj_val = obj_func(new_solution)

        # if the new solution is better, accept it
        if new_obj_val < initial_obj_val:
            initial_solution = new_solution.copy()
            initial_obj_val = new_obj_val
        else:
            # if the new solution is worse, accept it with a probability
            p = rd.random()
            if p < np.exp(-(new_obj_val - initial_obj_val) / T0):
                initial_solution = new_solution.copy()
                initial_obj_val = new_obj_val
    # print('old', initial_obj_val, ' new', new_obj_val)

    temp.append(T0)
    obj_val.append(initial_obj_val)
    T0 = ALPHA * T0

print('Final solution is valid:', validate_solution(initial_solution, TIME_WINDOW))
# print the last 50 objective function values vertically
for i in range(50):
    print(obj_val[-50 + i])
print('Final objective function value:', initial_obj_val)
# print scheduled times
for key, value in initial_solution.items():
    print(key.id_number, 'scheduled at', decimal_to_time(value))

plt.plot(temp, obj_val)
plt.xlabel('Temperature')
plt.ylabel('Objective Value')
plt.title('Simulated Annealing')
plt.xlim(T0, 0)
plt.xticks(np.arange(min(temp), max(temp)))
plt.show()
