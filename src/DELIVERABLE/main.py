
'''Ensure that lower bounds in job_shop_problem.py under custom_objective function are adjusted according to dataset size'''

import os
import time
from src.production_orders import parse_data
from pso import PSO, ObjectiveFunction
from job_shop_problem import JobShopProblem, ScheduleError  # Assuming this is the correct import for JobShopProblem

def get_file_path():
    while True:
        file_path = input("Please enter the full path to the dataset file: ").strip()
        if os.path.exists(file_path):
            return file_path
        else:
            print(f"Error: The file '{file_path}' does not exist. Please try again.")

def get_objective_function():
    print("Choose an objective function:")
    print("0 - Custom Objective")
    print("1 - Makespan")
    print("2 - Classical Tardiness")
    while True:
        try:
            choice = int(input("Enter the number of your choice: ").strip())
            if 0 <= choice <= 2:
                return ObjectiveFunction(choice)
            else:
                print("Invalid choice. Please enter a number between 0 and 2.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 2.")

def main():
    data_path = get_file_path()
    objective_function = get_objective_function()

    file_name = os.path.basename(data_path)
    if file_name.endswith('_large'):
        num_particles = 200
        max_iter = 400
        phi1 = 2.5
        phi2 = 2.0
    elif file_name.endswith('_small'):
        num_particles = 50
        max_iter = 100
        phi1 = 2.0
        phi2 = 1.5
    else:
        num_particles = 100
        max_iter = 200
        phi1 = 2.0
        phi2 = 1.5

    data = parse_data(data_path)
    jssp = JobShopProblem.from_data(data)
    pso = PSO(jssp, num_particles=num_particles, max_iter=max_iter, phi1=phi1, phi2=phi2, objective_function=objective_function)
    best_solution, runtime = pso.iterate()
    schedule = pso.get_best_schedule()
    score = pso.evaluate(best_solution)
    print(f"Objective: {objective_function.name}, Score: {score}, Runtime: {runtime} seconds")
    jssp.visualize_schedule(schedule)

if __name__ == "__main__":
    main()

