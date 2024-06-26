### Structure
1. src/Akshat PSO -> contains all necessary python files to run PSO and CPLEX algorithms
2. src/Akshat PSO/exact_solution.py -> Contains Code for MIP formulation solved with CPLEX
3. src/Akshat PSO/job_shop_problem.py -> Contains the code for loading the Job Shop Problem instance
4. src/Akshat PSO/pso.py -> Contains the code for the PSO algorithm
5. src/Akshat PSO/main.py -> Main script to run the PSO algorithm


### Dependencies and Prerequisites
Make sure that the latest version of Python is installed (3.12) and make sure that all required packages are installed, this can be done through poetry: 

1. Make sure you have [poetry](https://python-poetry.org/docs/#installation) installed. The most simple way is to install with `pip install poetry`, but follow the instructions in the link. You can test if you have poetry installed by running `poetry --version`.
2. Open a terminal in the `./src` directory.
3. Run `poetry install` to download the dependencies.
4. Run `poetry shell` to set your terminal to use the newly created virtual environment.
5. To run a script, simply run `python3 path_to_file_to_run.py` in the terminal in which you started the new shell. Alternatively, if you are running from a notebook, point the kernel to the path of the newly created `.venv` folder in `./src/.venv/Scripts/python`.

### PSO Algorithm 

To ensure that the PSO algorithm runs correctly, the name of the dataset must end with:
1. _large -> Indicates a large dataset is being used, therefore PSO will use different parameters
2. _small -> Indicates a small dataset is being used
3. Every else will be considered to represent the original dataset

Additionally one needs to make sure that the lower bounds used to calculate the custom objective function are adjusted according to the size of the dataset. This must be done in the job_shop_problem.py code under the custom_objective function: 

    def custom_objective(self, schedule: schedule_type) -> float:
        """Calculate the custom objective function of the schedule.

        Args:
            schedule (schedule_type): schedule to be evaluated

        Returns:
            float: result of the evaluation
        """
        tardiness = self.boolean_tardiness(schedule)
        total_setup_time = self.total_setup_time(schedule)
        makespan = self.makespan(schedule)

        if self.LOW_TARDINESS is None:

            # FOR A NORMAL DATASET
            self.LOW_TARDINESS = 8.0

            # FOR THE SMALL DATASET
            # self.LOW_TARDINESS = 0.1

            # FOR LARGE DATASET
            # self.LOW_TARDINESS = 0.6 * tardiness

        if self.LOW_TOTAL_SETUP_TIME is None:

            # FOR A NORMAL DATASET
            # self.LOW_TOTAL_SETUP_TIME = 95.0

            # FOR A SMALL DATASET
            # self.LOW_TOTAL_SETUP_TIME = 35.0

            # FOR A LARGE DATASET
            self.LOW_TOTAL_SETUP_TIME = 0.6 * total_setup_time

        if self.LOW_MAKESPAN is None:

            # FOR A NORMAL DATASET
            # self.LOW_MAKESPAN = 3502.0

            # FOR A SMALL DATASET
            # self.LOW_MAKESPAN = 2279.0

            # FOR A LARGE DATASET
            self.LOW_MAKESPAN = 0.6 * makespan

        return (
            (tardiness - self.LOW_TARDINESS) / self.LOW_TARDINESS
            + (total_setup_time - self.LOW_TOTAL_SETUP_TIME) / self.LOW_TOTAL_SETUP_TIME
            + (makespan - self.LOW_MAKESPAN) / self.LOW_MAKESPAN
        )

In the code above, since the LOW_MAKESPAN, LOW_TARDINESS, AND LOW_TOTAL_SETUP_TIME under the "FOR A LARGE DATASET" is not commented out, suggesting that these values will be used as the lower bounds during the custom objective function calcualtion. When using the smaller or orginally sized dataset, ensure that the appropriate value is active and not commented out. 


### Running the PSO
1. Make dependencies have been installed using poetry
2. Run job_shop_problem.py python file 
3. Run pso.py python file
4. Run main.py

To change the parameter settings of the PSO to explore differet solutions, one can change these under main.py python file. 3 sets of parameters settings are defined here, one for each of the different sizes of datasets.

### Exact Method
Install a CPLEX solver, visit https://www.ibm.com/products/ilog-cplex-optimization-studio, to install. The exact method resides in src/schedule_generator/exact_solution.py. This model struggles with bigger magnitudes of production orders, and are only really able to solve problems up to around 7 produciton orders. In order to solve it its highly recomended to have access to a strong solver such as CPLEX (which is also the only officially tested solver), since the mathematical model is rather complex. Below is an example on how it could be solved with CPLEX installed on your system.

    from src.schedule_generator.main import JobShopProblem
    from src.production_orders import parse_data
    from src.exact_solution import generate_model, solve_model, get_schedule

    # Load the problem
    jssp = JobShopProblem.from_problem(parse_data("path/to/data"))

    # Generate a pyomo concrete model
    model = generate_model(jssp)

    # Solve the model with CPLEX, you can also provide a time limit
    solve_model(model)

    # (Optional) visualize the schedule
    sc = get_schedule(model, jssp)
    jssp.visualize_schedule(sc)