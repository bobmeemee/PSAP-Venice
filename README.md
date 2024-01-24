# The In Port Scheduling and Assignment Problem

The in port scheduling and assignment problem (IPSAP) is a problem that 
arises in the scheduling of ships in a port. We consider data generated
based on the port of Venice, Italy.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)


## Project Overview

The in port scheduling and assignment problem (PSAP) is a problem that 
arises in the scheduling of container ships in a port. The problem is to 
assign berthing positions to incoming ships and to schedule the loading
and unloading of containers to and from the ships. The objective is to 
minimize the total time required to service all ships. The problem is 
complicated by the fact that the berthing positions are limited and 
that the loading and unloading of containers is subject to precedence 
constraints.


This project explores how machine learning (ML) can enhance problem-solving in the PSAP 
using metaheuristics. We compare the performance of different metaheuristics with and 
without ML assistance. ML helps predict the best metaheuristic parameters for a given 
problem, a process known as parameter tuning. We use a neural network (NN) for the ML tasks.

## Features

Highlight key features of your project. Use bullet points for clarity.

This project includes the following features:

- Metaheuristic algorithms to solve the PSAP and generate data for the ML-PSAP
- A neural network to predict the best metaheuristic parameters for a given problem
- Files to compare the performance of different metaheuristics with and without ML assistance

## Structure

The project is structured as follows:
- `data` contains the data files from the port of Venice, Italy
- `psap` contains the metaheuristic algorithms to solve the PSAP and generate data for the ML-PSAP
- `mlpsap` contains the neural network to predict the best metaheuristic parameters for a given problem
- `results` contains the files to compare the performance of different metaheuristics with and without ML assistance

## Installation

Clone the repository and install the required packages.
```bash
# Clone the repository
git clone

# Change the working directory to PSAP
cd PSAP

# Example installation command or steps
pip install -r requirements.txt
```


## Usage

An example of how to use the PSAP metaheuristic algorithms is shown below.

```python

from PSAP.Problem import prepare_movements, SolutionGenerator
from PSAP.PSAP_TABU import TabuSearch

# import data and prepare the movements
movements = prepare_movements(instance=1)

# initialize the solver
solver = TabuSearch(max_time=60, epochs=1000,
                    tabu_list_size=10, number_of_tweaks=10, affected_movements=10)

# initialize the solution generator
solutionGenerator = SolutionGenerator(movements, solver)

# generate a solution
solution, obj_val, prev_initial_solution = solutionGenerator.generate_solution()
```