# Code Source
This folder contains the code for the thesis; the test, benchmark, model, and heuristic code.

## ACO - Ant Colony Optimisation
For a minimal version of the ACO see the code below.
```python
from src.production_orders import parse_data
from src.schedule_generator.main import JobShopProblem, ObjectiveFunction
from src.schedule_generator.ant_colony_optimisation import TwoStageACO

# Load the problem
jssp = JobShopProblem.from_data(parse_data("path/to/excel/sheet"))

# Instantiate the ACO with the objective of minimising the makespan
aco = TwoStageACO(jssp, ObjectiveFunction.MAKESPAN)

# Run the Ant Colony for some generations
aco.run()

# Optional - Visualise the schedule
schedule = jssp.make_schedule_from_parallel(aco.best_solution[1])
jssp.visualize(schedule)
```

The ACO has some parameters that are important to consider when running it in order to achieve the best results. Some advice is to use the parallel with stock schedule generator since that one generally gives better solution. Furthermore, setting `tau_zero` as a rough estimate for the objective function is important so that the model does not have too small float numbers, and also to achieve better results. Increasing the number of ants does not necessarily provide better results, as there is a diminishing return. Lastly setting the convergence rate is important as to not land in a local optimal. Depending on the problem it could be interesting to use local search, however for the provided data sets this has not proven to be useful.

## Structure
* `src/examples/` contains three data sets for the thesis. These are small (5 orders), medium (7 orders), and large (100 orders) in size.
* `src/schedule_generator/` contains code for loading a `JobShopProblem` instance and the code for the Ant Colony Optimisation.
* `src/production_orders.py` contains the code for loading the data sets into `pydantic` and pandas DataFrame objects.
* `src/tests/` contains tests for the `production_orders.py` code.
