import random as rd
import pandas as pd


class Movement:
    def __init__(self, id_number, vessel_type, optimal_time):
        self.id_number = id_number
        self.optimal_time = time_to_decimal(optimal_time)
        self.scheduled_time = self.optimal_time
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


def read_data(instance):
    # read in the data
    df_movimenti = pd.read_excel(str('data/Instance_' + str(instance) + '.xlsx'), header=0, sheet_name='movimenti')
    df_precedenze = pd.read_excel(str('data/Instance_' + str(instance) + '.xlsx'), header=0, sheet_name='Precedenze')
    return df_movimenti, df_precedenze


# create a dictionary with the initial solution {movement: scheduled_time}
def generate_initial_solution(df, df_headway, deviation_scale=1, time_interval=5):
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
            if not (precedence_allowed == 1 and headway_time < 0):
                m.add_headway(other_mov, [precedence_allowed, headway_time])
            if j == len(df_headway) - 1:
                break
            j += 1

        init_sol[m] = scheduled_time
    return init_sol


# generate a neighboring solution by applying a random time deviation to a random set of movements
def generate_neighbor_solution(initial_solution, affected_movements, deviation_scale, time_interval=5):
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

        # choose a random time interval, non-gaussian
        # time_deviation = rd.choice([-1, 1]) * rd.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

        # apply the time deviation
        initial_solution[movement] += time_deviation / 60
        chosen_neighbors.append(movement)

    return initial_solution


def obj_func(solution, precedence=None):
    cost = 0
    for key, value in solution.items():
        # penalty for deviation from optimal time
        if key.vessel_type == 'Cargo ship':
            cost += 1 * abs(key.optimal_time - value) * 3.5
        else:
            cost += 1 * abs(key.optimal_time - value) * 3.5

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
                    if delta_t < key.headway.get(key2.id_number)[1] and delta_t > 0.:
                        cost += abs(delta_t) * 7
                else:
                    # no headway has to be applied
                    cost += 0

        # check if the precedence constraints are satisfied
        if precedence is not None:
            precedences = precedence.get(key)  # {m': headway}
            if precedences is None:
                continue
            for other_movement, headway in precedences.items():
                if key == other_movement:
                    print('ERROR: Movement is in its own precedence constraints')
                    return False

                # precedence can be negative
                time_difference = solution[other_movement] - value
                if headway >= 0:
                    if time_difference < headway:
                        cost += 5 * abs(time_difference - headway)

                else:
                    if time_difference > headway:
                        cost += 5 * abs(time_difference - headway)

    return cost


def validate_solution(solution, time_window, print_errors=False):
    if solution is None:
        print('ERROR: No solution found' if print_errors else '')
        return False
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
                    # only check the headway if the next movement is after the first movement
                    delta_t = value2 - value
                    if delta_t < key.headway.get(key2.id_number)[1] and delta_t > 0.:
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


def validate_precedence_constraints(solution: dict, precedence: dict):
    if precedence is None:
        return True
    # check if the precedence constraints are satisfied
    for movement, time in solution.items():
        precedences = precedence[movement]
        for other_movement, headway in precedences.items():
            if movement == other_movement:
                print('ERROR: Movement is in its own precedence constraints')
                return False

            # precedence can be negative
            time_difference = solution[other_movement] - time
            if headway >= 0:
                if time_difference < headway:
                    return False

            else:
                if time_difference > headway:
                    return False

    return True


def earliest(solution: dict, movements: list):
    earliest_movement = None
    earliest_time = 1000000
    for m in movements:
        if solution.get(m) < earliest_time:
            earliest_time = solution.get(m)
            earliest_movement = m
    return earliest_movement, earliest_time
