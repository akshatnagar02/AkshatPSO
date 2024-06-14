# Code Source
This folder contains the code for the thesis; the test, benchmark, model, and heuristic code.

## Structure
* `src/examples/` contains three data sets for the thesis. These are small (5 orders), medium (7 orders), and large (100 orders) in size.
* `src/schedule_generator/` contains code for loading a `JobShopScheduling` instance and the code for the Ant Colony Optimisation.
* `src/production_orders.py` contains the code for loading the data sets into `pydantic` and pandas DataFrame objects.
* `src/tests/` contains tests for the `production_orders.py` code.
