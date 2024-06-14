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

# Instantiate the ACO
aco = TwoStageACO(jssp, ObjectiveFunction.MAKESPAN)

# Run the Ant Colony for some generations
aco.run()

# Optional - Visualise the schedule
schedule = jssp.make_schedule_from_parallel(aco.best_solution[1])
jssp.visualize(schedule)
```


## Structure
* `src/examples/` contains three data sets for the thesis. These are small (5 orders), medium (7 orders), and large (100 orders) in size.
* `src/schedule_generator/` contains code for loading a `JobShopProblem` instance and the code for the Ant Colony Optimisation.
* `src/production_orders.py` contains the code for loading the data sets into `pydantic` and pandas DataFrame objects.
* `src/tests/` contains tests for the `production_orders.py` code.
