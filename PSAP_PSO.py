import pandas as pd
import time
import random as rd
import numpy as np
from matplotlib import pyplot as plt

from Problem import validate_solution, generate_initial_solution, read_data, earliest

TIME_INTERVAL = 5
TIME_WINDOW = 60 * 6


class Particle:
    def __init__(self, position=None):
        self.position = position  # dictionary with movements as keys and scheduled times as values
        self.best_position = position  # dictionary with movements as keys and scheduled times as values of the best
        # solution found by the particle)

        self.velocity = {mov: 0 for mov in position}  # dictionary with movements as keys and time units as values

        self.informants = []
        self.best_informant = None

    def update_best_position(self):
        if obj_func(self.position) < obj_func(self.best_position):
            self.best_position = self.position.copy()

    def set_informants(self, swarm, percentage):
        self.informants = [self]
        while len(self.informants) < int(percentage * len(swarm)):
            next_informant = swarm[rd.randint(0, len(swarm) - 1)]
            while next_informant in self.informants:
                next_informant = swarm[rd.randint(0, len(swarm) - 1)]
            self.informants.append(next_informant)

        self.best_informant = self.informants[
            np.argmin([obj_func(informant.position) for informant in self.informants])]

    def get_previous_best_informant(self):
        return self.best_informant

    def set_best_informant(self):
        self.best_informant = self.informants[
            np.argmin([obj_func(informant.position) for informant in self.informants])]

    def update_velocity(self, alltime_best_solution, alpha, beta, gamma, delta):
        # get random numbers from 0 to corresponding coefficient
        b = rd.random()
        c = rd.random()
        d = rd.random()
        # get new velocity
        new_velocity = {}
        for m in self.velocity:
            new_velocity[m] = alpha * self.velocity[m] + \
                              beta * b * (self.best_position[m] - self.position[m]) + \
                              gamma * c * (self.best_informant.position[m] - self.position[m]) + \
                              delta * d * (alltime_best_solution[m] - self.position[m])
        self.velocity = new_velocity

    def move_position(self, epsilon):
        # get new position
        new_position = {}
        for m in self.position:
            new_position[m] = self.position[m] + epsilon * self.velocity[m]
        self.position = new_position


def create_swarm(movements, swarm_size, deviation_scale=45, time_interval=5):
    swarm = []
    for i in range(swarm_size):
        # create a random solution
        for m in movements:
            time_deviation = round(rd.gauss(0, deviation_scale) / time_interval) * time_interval
            m.set_scheduled_time(m.optimal_time + time_deviation / 60)
        random_solution = {m: m.get_scheduled_time() for m in movements}

        # create a particle with the random solution as position
        particle = Particle(position=random_solution.copy())
        swarm.append(particle)
    return swarm


def obj_func(solution, precedence=None):
    cost = 0
    for key, value in solution.items():
        # penalty for deviation from optimal time
        if key.vessel_type == 'Cargo ship':
            cost += 1 * abs(key.optimal_time - value) * 5
        else:
            cost += 1 * abs(key.optimal_time - value) * 5

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
                        cost += abs(delta_t) * 2000
                        if abs(delta_t) * 2000 < 500:
                            cost += 5000
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
                        cost += 10 * abs(time_difference - headway)

                else:
                    if time_difference > headway:
                        cost += 10 * abs(time_difference - headway)

    return cost


# ============PARITCLE SWARM OPTIMIZATION================
# MILP SOLVER BASED ON PSO
def solve_with_precedence_constraints_PSO(movements: list, precedence: dict, max_time: int, epochs, swarm_size=10,
                                          informant_percentage=0.5,
                                          alpha=0.95, beta=0.8, gamma=0.8, delta=0.8, epsilon=0.8,
                                          vessel_time_window=60 * 6):
    # the initial solution is a dictionary with movements as keys and scheduled times as values
    # the precedence constraints are a dictionary with movements as keys and lists of movements as values
    # the max_time is the maximum time in minutes that the movements can be scheduled

    # start the timer
    start_time = time.time()

    # create the swarm
    swarm = create_swarm(movements, swarm_size)

    # set the informants
    for particle in swarm:
        particle.set_informants(swarm, informant_percentage)

    # set the all time best solution
    all_time_best_solution = {}
    all_time_best_obj_val = float('inf')
    for particle in swarm:
        if obj_func(particle.best_position, precedence) < all_time_best_obj_val:
            all_time_best_obj_val = obj_func(particle.best_position, precedence)
            all_time_best_solution = particle.best_position.copy()

    e = 0
    alltimebestsolutions = []
    alltimebest_epoch = []

    if validate_solution(all_time_best_solution, vessel_time_window, print_errors=False):
        return all_time_best_solution, all_time_best_obj_val

    # while the time limit is not reached
    while time.time() - start_time < max_time and e < epochs and not validate_solution(all_time_best_solution,
                                                                                       vessel_time_window):

        # if e >= epochs:
        #     initial_solution = {m: m.optimal_time for m in movements}
        #     initial_obj_val = obj_func(initial_solution)
        #     e = 0
        #     attempt += 1

        for particle in swarm:
            particle.update_velocity(all_time_best_solution, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
            particle.move_position(epsilon=epsilon)

        for particle in swarm:
            particle.update_best_position()
            particle.set_best_informant()
            particle_obj_val = obj_func(particle.best_position, precedence)
            if particle_obj_val < all_time_best_obj_val:
                all_time_best_solution = particle.best_position.copy()
                all_time_best_obj_val = particle_obj_val

            # # if the new solution is better, accept it
            # if particle_obj_val < all_time_best_obj_val:
            #     all_time_best_solution = particle.best_position.copy()
            #     all_time_best_obj_val = particle_obj_val
            # # if the new solution is worse, accept it with a probability
            # else:
            #     # if the new solution is worse, accept it with a probability
            #     p = rd.random()
            #     if p < np.exp(-(particle_obj_val - all_time_best_obj_val) / t0):
            #         all_time_best_solution = particle.best_position.copy()
            #         all_time_best_obj_val = particle_obj_val

        alltimebestsolutions.append(all_time_best_obj_val)
        alltimebest_epoch.append(e)
        e += 1

    # plot the convergence with
    if len(alltimebestsolutions) > 0:
        plt.plot(alltimebest_epoch, alltimebestsolutions)
        print(alltimebest_epoch)
        plt.xlabel('Epochs --with lenght movements: ' + str(len(movements)))
        plt.ylabel('Objective value')
        plt.show()

    if validate_solution(all_time_best_solution, vessel_time_window, print_errors=True):
        return all_time_best_solution, all_time_best_obj_val
    else:
        print("No solution, objective value: ", all_time_best_obj_val)
        return None, None


def solution_generating_procedure(movements: list, l, t, epochs=200, swarm_size=10, informant_percentage=0.5,
                                  alpha=0.95, beta=0.8, gamma=0.8, delta=0.8, epsilon=0.8):
    # movements is a list of movements
    # sort the movements by time
    sorted_movements = sorted(movements, key=lambda x: x.optimal_time)

    # select the first l movements
    movements_subset = sorted_movements[:l]

    fixed_movements = []
    precedence = {}
    solution = {}
    while len(solution) != len(movements):
        # unite the new subset with the fixed movements
        problem_subset = fixed_movements + movements_subset
        # solve the problem with the new subset and the precedence constraints
        solution, obj_val = solve_with_precedence_constraints_PSO(problem_subset, precedence, t, epochs=epochs,
                                                                  informant_percentage=informant_percentage,
                                                                  swarm_size=swarm_size,
                                                                  alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                                                  epsilon=epsilon)

        # if no solution was found, return None
        if solution is None:
            print("No solution found while using precedence constraints")
            print("reached: ", len(fixed_movements), "/", len(movements))
            return None, None
        else:
            # get the earliest movement in the solution that is not in the fixed movements
            first_movement, first_movement_time = earliest(solution, movements_subset)
            # append the precedence constraints {first_movement: {m_i: time_difference_i}}
            first_movement_precedences = {}
            for m, time in solution.items():
                if m != first_movement:
                    difference = time - first_movement_time
                    first_movement_precedences[m] = difference
            precedence[first_movement] = first_movement_precedences

            # append the movement to the fixed movements
            fixed_movements.append(first_movement)

        # remove the first movement from the candidates for the subset and select the next l earliest movements
        sorted_movements.remove(first_movement)
        movements_subset = sorted_movements[:l]

    if len(solution) == len(movements):
        return solution, obj_val
    else:
        return None, None


def generate_parameters_PSO(epochs_rng, swarm_size_rng, informant_percentage_rng, alpha_rng, beta_rng, gamma_rng,
                            delta_rng, epsilon_rng):
    epochs = rd.randint(epochs_rng[0], epochs_rng[1])
    swarm_size = rd.randint(swarm_size_rng[0], swarm_size_rng[1])
    informant_percentage = rd.uniform(informant_percentage_rng[0], informant_percentage_rng[1])
    alpha = rd.uniform(alpha_rng[0], alpha_rng[1])
    beta = rd.uniform(beta_rng[0], beta_rng[1])
    gamma = rd.uniform(gamma_rng[0], gamma_rng[1])
    delta = rd.uniform(delta_rng[0], delta_rng[1])
    epsilon = rd.uniform(epsilon_rng[0], epsilon_rng[1])

    return epochs, swarm_size, informant_percentage, alpha, beta, gamma, delta, epsilon


if __name__ == '__main__':
    sol_found = False
    instance = 2
    valid_solutions = []
    objective_values = []

    df = pd.DataFrame(columns=['instance', 'number of movements', 'median delay', 'average delay', 'epochs', 'obj_val',
                               'neighborhood_size', 't0', 'alpha', 'neighbor_deviation_scale', 'affected_movements',
                               'time_interval', 'vessel_time_window'])

    df_instance = pd.DataFrame(columns=['instance', 'number_of_movements', 'number_of_vessels', 'average_headway',
                                        'std_dev_headway', 'spread', 'average_time_between_movements',
                                        'average_travel_time'])

    time_window = 60 * 6
    time_interval = 5
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
        result_list = [elem for index, elem in enumerate(sorted_movements, 1) if index % 2 != 0]
        print("Objective value initial solution: ", obj_func(initial_solution))

        # run the solution generating procedure 10 times for each instance and save the results
        for _ in range(3):
            epochs, swarm_size, informant_percentage, alpha, beta, gamma, delta, epsilon = \
                generate_parameters_PSO(epochs_rng=[300, 300],
                                        swarm_size_rng=[100, 100],
                                        informant_percentage_rng=[0.5, .5],
                                        alpha_rng=[0.7, 0.9],
                                        beta_rng=[0.5, 0.7],
                                        gamma_rng=[0.5, 0.7],
                                        delta_rng=[0.4, 0.5],
                                        epsilon_rng=[0.2, 0.4])
            print("epochs: ", epochs, "swarm_size: ", swarm_size, "informant_percentage: ", informant_percentage,
                  "alpha: ", alpha, "beta: ", beta, "gamma: ", gamma, "delta: ", delta, "epsilon: ", epsilon)

            initial_solution, obj_val = solution_generating_procedure(result_list, 3, 20, epochs=100,
                                                                      swarm_size=150,
                                                                      informant_percentage=informant_percentage,
                                                                      alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                                                                      epsilon=epsilon)

            valid_solution_params = []
            if initial_solution is not None:
                # set the movement scheduled to the result of the solution generating procedure
                for m, t in initial_solution.items():
                    m.set_scheduled_time(t)
                print("Solution", _, " found for instance", instance, "(", len(initial_solution), ")")
                obj_val = obj_func(initial_solution)
                print("Objective value: ", obj_val)
                valid_solution_params.append([epochs, swarm_size, informant_percentage, alpha, beta, gamma, delta,
                                              epsilon])

                # df.loc[len(df.index)] = [instance, len(initial_solution),
                #                          np.median([abs(m.get_delay()) for m in initial_solution.keys()]),
                #                          np.mean([abs(m.get_delay()) for m in initial_solution.keys()]),
                #                          epochs, obj_val, neighborhood_size, t0, alpha, neighbor_deviation_scale,
                #                          affected_movements, TIME_INTERVAL, TIME_WINDOW]
            else:
                print("No solution found for instance", instance)
                # df.loc[len(df.index)] = [instance, None, None, None, epochs, None, neighborhood_size, t0, alpha,
                #                          neighbor_deviation_scale, affected_movements, TIME_INTERVAL, TIME_WINDOW]

        instance += 1
        print(valid_solution_params)

        try:
            df.to_excel('results/PSO/output.xlsx', index=False)
        except PermissionError:
            print("Please close the file output.xlsx and try again")

        print("solutions found: ", len(df.index))
