from PSAP.Problem import SolutionGenerator, prepare_movements
from PSAP.PSAP_TABU import TabuSearch

# main funciton

if __name__ == '__main__':
    movements = prepare_movements(instance=1)

    solver = TabuSearch(max_time=60, epochs=1000,
                        tabu_list_size=10, number_of_tweaks=10, affected_movements=10)

    solutionGenerator = SolutionGenerator(movements, solver)
    initial_solution, obj_val, prev_initial_solution = solutionGenerator.generate_solution()

    print(initial_solution)
    print(obj_val)
    print(prev_initial_solution)
