import pandas as pd
from Problem import collect_instance_data, read_data, generate_initial_solution

TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6

df_instance = pd.DataFrame(columns=['instance', 'number_of_movements', 'number_of_headways',
                                    'number_of_vessels', 'average_headway',
                                    'std_dev_headway', 'spread', 'average_time_between_movements',
                                    'average_travel_time'])

# avg travel distance?
# number of berths in same interval?

for instance in range(1, 101):
    print("Solving instance " + str(instance))

    df_movimenti, df_precedenze, df_tempi = read_data(instance)
    # generate the initial solution
    initial_solution = generate_initial_solution(df=df_movimenti, df_headway=df_precedenze, deviation_scale=1,
                                                 time_interval=TIME_INTERVAL, df_tempi=df_tempi)

    movements = list(initial_solution.keys())
    sorted_movements = sorted(movements, key=lambda x: x.optimal_time)

    for idx in range(2):
        if idx == 0:
            result_list = [elem for index, elem in enumerate(sorted_movements, 1) if index % 2 != 0]
            plus = 0
        else:
            result_list = [elem for index, elem in enumerate(sorted_movements, 1) if index % 2 == 0]
            plus = 100

        instance_data = collect_instance_data(result_list)
        df_instance.loc[len(df_instance.index)] = [instance + plus, instance_data['number_of_movements'],
                                                   instance_data['number_of_headways'],
                                                   instance_data['number_of_vessels'], instance_data['average_headway'],
                                                   instance_data['std_dev_headway'], instance_data['spread'],
                                                   instance_data['average_time_between_movements'],
                                                   instance_data['average_travel_time']]

        try:
            df_instance.to_excel('results/instance200.xlsx', index=False)
        except PermissionError:
            print("Please close the file instance_data.xlsx and try again")
