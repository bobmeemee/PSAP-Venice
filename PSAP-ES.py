import pandas as pd
import random as rd
import numpy as np

from Problem import Movement, time_to_decimal, decimal_to_time, validate_solution, generate_initial_solution

# ============EVOLUTIONARY ALGORITHM================

# ======================================================================================================================
# PARAMETERS
# ======================================================================================================================
INSTANCE = 1
POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.9
TOURNAMENT_SIZE = 2




# read in the data
df_movimenti = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='movimenti')
df_precedenze = pd.read_excel(str('data/Instance_' + str(INSTANCE) + '.xlsx'), header=0, sheet_name='Precedenze')



