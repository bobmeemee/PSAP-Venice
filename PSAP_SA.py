import numpy as np
import pandas as pd
import random as rd

from matplotlib import pyplot as plt

INSTANCE = 11
TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6

T0 = 100
ALPHA = 0.98
NEIGHBORHOOD_SIZE = 10  # number of neighbours to consider at each iteration
EPOCHS = 500  # number of iterations


class Movement:
    def __init__(self, id_number, vessel_type, optimal_time):
        self.id_number = id_number
        self.optimal_time = time_to_decimal(optimal_time)
        # self.scheduled_time = self.optimal_time
        self.vessel_type = vessel_type
        self.headway = dict()

    def add_headway(self, id_number, time):
        self.headway[id_number] = time

    # print the movement
    def __str__(self):
        # print the headway in a readable way
        hw = ''
        for key, value in self.headway.items():
            hw += str(key) + ': [' + str(value[0]) + '-' + str(decimal_to_time(value[1])) + '] ==|== '

        return str(self.id_number) + ' ' + str(self.vessel_type) + ' ' + str(decimal_to_time(self.optimal_time)) + ' ' \
            + hw


def time_to_decimal(time):
    hour = int(time.split(':')[0])
    minute = float(time.split(':')[1])
    return hour + minute / 60


def decimal_to_time(decimal):
    hour = int(decimal)
    minute = round(float((decimal - hour) * 60))
    return str(hour) + ':' + str(minute)


# read in the data
df_movimenti = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='movimenti')
df_precedenze = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='Precedenze')


# create a dictionary with the initial solution
def generate_initial_solution(df, df_headway, deviation_scale, time_interval=5):
    init_sol = dict()
    j = 0
    for i in range(len(df)):
        m = Movement(df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 5])
        # add a random amount of minutes to the optimal time in intervals of TIME_INTERVAL
        # the amount of minutes is drawn from a normal distribution with mean 0 and standard deviation deviation_scale
        # this deviation scale is a parameter that can be tuned
        time_deviation = round(rd.gauss(0, deviation_scale) / time_interval) * time_interval
        scheduled_time = m.optimal_time + time_deviation / 60

        # add the headways to the movement
        while df_headway.iloc[j, 0] == m.id_number:
            precedence_allowed = df_headway.iloc[j, 1]
            other_mov = df_headway.iloc[j, 2]
            headway_time = df_headway.iloc[j, 3] / 60  # convert to decimal
            m.add_headway(other_mov, [precedence_allowed, headway_time])
            if j == len(df_headway) - 1:
                break
            j += 1

        init_sol[m] = scheduled_time
    return init_sol


def generate_neighbor_solution(initial_solution, affected_movements, deviation_scale, time_interval=5):
    chosen_neighbors = []
    for _ in range(affected_movements):
        # choose a random movement
        movement = rd.choice(list(initial_solution.keys()))
        while movement in chosen_neighbors:
            movement = rd.choice(list(initial_solution.keys()))
        # choose a random time interval
        time_deviation = round(rd.gauss(0, deviation_scale) / time_interval) * time_interval

        # choose a random time interval, non-gaussian
        # time_deviation = rd.choice([-1, 1]) * rd.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

        # apply the time deviation
        initial_solution[movement] += time_deviation / 60
        chosen_neighbors.append(movement)

    return initial_solution


def obj_func(solution):
    cost = 0
    for key, value in solution.items():
        # penalty for deviation from optimal time
        if key.vessel_type == 'Cargo ship':
            cost += 5 * abs(key.optimal_time - value) * 3.5
        else:
            cost += 10 * abs(key.optimal_time - value) * 3.5

        # if abs(key.optimal_time - value) > TIME_WINDOW / 60 / 2:
        #     cost += 100 * abs(key.optimal_time - value)

        # penalty for headway violations
        # TODO: tweak the penalty values
        for key2, value2 in solution.items():
            if key != key2:
                if key.headway.get(key2.id_number)[0] == 0:
                    # m and m' are the same vessel, m can't be scheduled before m'
                    delta_t = value2 - value
                    if delta_t < 0:
                        cost += 1 * abs(delta_t)
                elif key.headway.get(key2.id_number)[0] == 1:
                    # headway has to be applied
                    delta_t = value2 - value
                    if delta_t < key.headway.get(key2.id_number)[1]:
                        cost += abs(delta_t) * 1
                else:
                    # no headway has to be applied
                    cost += 0
    return cost


def validate_solution(solution, time_window, print_errors=False):
    errors = ''
    for key, value in solution.items():
        if abs(key.optimal_time - value) > time_window / 60 / 2:
            errors += ('Movement ' + str(key.id_number) + ' is scheduled outside its time window' + ' delta_t = ' +
                       str(abs(value - key.optimal_time) * 60) + '\n')

        for key2, value2 in solution.items():
            if key != key2:
                if key.headway.get(key2.id_number)[0] == 0:
                    # m and m' are the same vessel, m can't be scheduled before m'
                    delta_t = value2 - value
                    if delta_t < 0.:
                        errors += ('Movement ' + str(key.id_number) + ' is scheduled before movement ' +
                                   str(key2.id_number) + ' (same vessel)' + ' delta_t = ' + str(delta_t) + '\n')

                elif key.headway.get(key2.id_number)[0] == 1:
                    # headway has to be applied
                    delta_t = value2 - value
                    if delta_t < key.headway.get(key2.id_number)[1]:
                        errors += ('Movement ' + str(key.id_number) + ' is scheduled too close to movement ' +
                                   str(key2.id_number) + ' (headway)' + ' delta_t = ' + str(delta_t * 60) +
                                   ' required headway = ' + str(key.headway.get(key2.id_number)[1] * 60) + '\n')

    if errors.__len__() > 0 and print_errors:
        print(errors)
        return False
    elif errors.__len__() > 0:
        return False
    else:
        return True


# SIMULATED ANNEALING
def SA_solve(epochs, neighborhood_size, df_movimenti, df_precedenze, t0, alpha, time_window, time_interval=5,
             print_errors=False, initial_deviation_scale=10, neighbor_deviation_scale=10, affected_movements=3):
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
        # print('old', initial_obj_val, ' new', new_obj_val)

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
