import pandas as pd
import random as rd

TIME_INTERVAL = 5


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
            hw += str(key) + ': ' + str(value[0]) + '-' + str(decimal_to_time(value[1])) + '|'

        return str(self.id_number) + ' ' + str(self.vessel_type) + ' ' + str(decimal_to_time(self.optimal_time)) + ' ' \
            + hw


def time_to_decimal(time):
    hour = int(time.split(':')[0])
    minute = int(time.split(':')[1])
    return hour + minute / 60


def decimal_to_time(decimal):
    hour = int(decimal)
    minute = int((decimal - hour) * 60)
    return str(hour) + ':' + str(minute)


# read in the data
df_movimenti = pd.read_excel('data/Instance_4.xlsx', header=0, sheet_name='movimenti')
df_precedenze = pd.read_excel('data/Instance_4.xlsx', header=0, sheet_name='Precedenze')


# create a dictionary with the initial solution
def generate_initial_solution(df, df_headway, deviation_scale):
    init_sol = dict()
    j = 0
    for i in range(len(df)):
        m = Movement(df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 5])
        # add a random amount of minutes to the optimal time in intervals of TIME_INTERVAL
        # the amount of minutes is drawn from a normal distribution with mean 0 and standard deviation deviation_scale
        # this deviation scale is a parameter that can be tuned
        time_deviation = round(rd.gauss(0, deviation_scale) / TIME_INTERVAL) * TIME_INTERVAL
        scheduled_time = m.optimal_time + time_deviation / 60

        # add the headways to the movement
        print(df_headway.iloc[j, 0], m.id_number)
        while df_headway.iloc[j, 0] == m.id_number:
            precedence_allowed = df_headway.iloc[j, 1]
            other_mov = df_headway.iloc[j, 2]
            headway_time = df_headway.iloc[j, 3] / 60  # convert to decimal
            m.add_headway(other_mov, [precedence_allowed, headway_time])
            j += 1

        init_sol[m] = scheduled_time
    return init_sol



def obj_func(solution):
    cost = 0
    for key, value in solution.items():
        # penalty for deviation from optimal time
        if key.vessel_type == 'Cargo ship':
            cost += 5 * abs(key.optimal_time - value)
        else:
            cost += 10 * abs(key.optimal_time - value)

        # penalty for headway violations
        # TODO: tweak the penalty values
        for key2, value2 in solution.items():
            if key != key2:
                if key.headway.get(key2)[0] == 0:
                    # m and m' are the same vessel, m can't be scheduled before m'
                    delta_t = value2 - value
                    if delta_t < 0:
                        cost += 1000
                elif key.headway.get(key2)[0] == 1:
                    # headway has to be applied
                    delta_t = value2 - value
                    if delta_t < key.headway.get(key2)[1]:
                        cost += delta_t * 10
                else:
                    # no headway has to be applied
                    cost += 0
    return cost


initial_solution = generate_initial_solution(df_movimenti, df_precedenze, 1)
