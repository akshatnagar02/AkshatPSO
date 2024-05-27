"""Mathematical model for the exact solution of the scheduling problem."""

import pyomo.environ as pyo
from src.schedule_generator.main import JobShopProblem, schedule_type
from src.production_orders import parse_data
import numpy as np
from pyomo.util.infeasible import log_infeasible_constraints
import logging

logger = logging.Logger(__name__)  
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
logger.addHandler(ch)

H = 10e3


def generate_model(jssp: JobShopProblem) -> pyo.ConcreteModel:
    model = pyo.ConcreteModel()
    model.jobs = pyo.Set(initialize=range(len(jssp.jobs)))
    model.machines = pyo.Set(initialize=range(len(jssp.machines)))

    model.t = pyo.Var(model.jobs, domain=pyo.NonNegativeReals, initialize=0)
    model.alpha = pyo.Var(model.jobs, model.machines, domain=pyo.Binary, initialize=0)
    model.beta = pyo.Var(model.jobs, model.jobs, domain=pyo.Binary, initialize=0)
    model.tardiness = pyo.Var(model.jobs, domain=pyo.NonNegativeReals, initialize=0)
    model.sigma = pyo.Var(model.jobs, domain=pyo.Integers, initialize=0)
    model.mu = pyo.Var(model.jobs, domain=pyo.Integers, initialize=0)
    model.z = pyo.Var(model.jobs, bounds=(0, 1), initialize=0)
    model.w = pyo.Var(model.jobs, bounds=(0, 1), initialize=0)
    model.epsilon = pyo.Var(model.jobs, domain=pyo.NonNegativeReals, initialize=0)

    # Define objective function
    model.objective = pyo.Objective(
        expr=pyo.quicksum(model.tardiness[j] for j in model.jobs), sense=pyo.minimize
    )

    # Calculate tardiness for each job
    def calculate_tardiness(m, j):
        return (
            m.tardiness[j]
            >= m.t[j] + m.epsilon[j]
            + pyo.quicksum(
                m.alpha[j, machine] * jssp.jobs[j].available_machines.get(machine, 0)
                for machine in m.machines
                if machine in jssp.jobs[j].available_machines
            )
            - jssp.jobs[j].days_till_delivery * 24 * 60
        )

    model.calculate_tardiness = pyo.Constraint(model.jobs, rule=calculate_tardiness)

    # Only one machine assigned to each job
    def one_machine_per_job(m, j):
        return (
            pyo.quicksum(
                m.alpha[j, machine]
                for machine in m.machines
                if machine in jssp.jobs[j].available_machines
            )
            == 1
        )

    model.one_machine_per_job = pyo.Constraint(model.jobs, rule=one_machine_per_job)

    # Keep alpha 0 if job cannot be assigned to machine
    def alpha_zero_if_not_available(m, j, machine):
        if machine not in jssp.jobs[j].available_machines:
            return m.alpha[j, machine] == 0
        else:
            return pyo.Constraint.Skip
    
    model.alpha_zero_if_not_available = pyo.Constraint(model.jobs, model.machines, rule=alpha_zero_if_not_available)

    # Keep precedence order of jobs
    def precedence_order(m, j1, j2):
        if len(jssp.jobs[j1].dependencies) > 0 and j2 in jssp.jobs[j1].dependencies:
            return m.t[j1] >= m.t[j2] + m.epsilon[j2] + pyo.quicksum(
                m.alpha[j2, machine] * jssp.jobs[j2].available_machines.get(machine, 0)
                for machine in m.machines
                if machine in jssp.jobs[j2].available_machines
            )
        else:
            return pyo.Constraint.Skip

    model.precedence_order = pyo.Constraint(
        model.jobs, model.jobs, rule=precedence_order
    )

    # Make sure that jobs does not overlap
    def no_overlapping_1(m, j1, j2, machine):
        if (
            j1 == j2
            or machine not in jssp.jobs[j1].available_machines
            or machine not in jssp.jobs[j2].available_machines
        ):
            return pyo.Constraint.Skip
        else:
            return (
                m.t[j1]
                >= m.t[j2]
                + jssp.jobs[j2].available_machines[machine]
                + jssp.setup_times[j2][j1]
                + m.epsilon[j2]
                - (2 - m.alpha[j1, machine] - m.alpha[j2, machine] + m.beta[j1, j2]) * H
            )

    model.no_overlapping_1 = pyo.Constraint(
        model.jobs, model.jobs, model.machines, rule=no_overlapping_1
    )

    def no_overlapping_2(m, j1, j2, machine):
        if (
            j1 == j2
            or machine not in jssp.jobs[j1].available_machines
            or machine not in jssp.jobs[j2].available_machines
        ):
            return pyo.Constraint.Skip
        else:
            return (
                m.t[j2]
                >= m.t[j1]
                + jssp.jobs[j1].available_machines[machine]
                + jssp.setup_times[j1][j2]
                + m.epsilon[j1]
                - (3 - m.alpha[j1, machine] - m.alpha[j2, machine] - m.beta[j1, j2]) * H
            )

    model.no_overlapping_2 = pyo.Constraint(
        model.jobs, model.jobs, model.machines, rule=no_overlapping_2
    )

    # Floor function for start time day
    def floor_function_start(m, j):
        return m.t[j]/(24*60) == m.sigma[j] + m.z[j]
    
    model.floor_function_start = pyo.Constraint(model.jobs, rule=floor_function_start)

    # Enforce start time is after start of machine
    def enforce_start_time(m, j, machine):
        if machine not in jssp.jobs[j].available_machines:
            return pyo.Constraint.Skip
        else:
            return m.t[j] >= 24*60 * m.sigma[j] + jssp.machines[machine].start_time - (1-m.alpha[j, machine]) * H
        
    model.enforce_start_time = pyo.Constraint(model.jobs, model.machines, rule=enforce_start_time)

    # Enforce start time is before end of machine
    def enforce_end_time(m, j, machine):
        if machine not in jssp.jobs[j].available_machines:
            return pyo.Constraint.Skip
        else:
            return m.t[j] <= 24*60 * m.sigma[j] + jssp.machines[machine].end_time + (1-m.alpha[j, machine]) * H
        
    model.enforce_end_time = pyo.Constraint(model.jobs, model.machines, rule=enforce_end_time)

    # Floor function for end time day
    def floor_function_end(m, j):
        return (m.t[j]+pyo.quicksum(m.alpha[j, machine] * (jssp.jobs[j].available_machines[machine] - jssp.machines[machine].end_time) for machine in m.machines if machine in jssp.jobs[j].available_machines))/(24*60) == m.mu[j] + m.w[j]

    model.floor_function_end = pyo.Constraint(model.jobs, rule=floor_function_end)

    # Add extra time for downtime
    def add_extra_time(m, j, machine):
        if machine not in jssp.jobs[j].available_machines:
            return pyo.Constraint.Skip
        else:
            return m.t[j] + m.epsilon[j] >= 24*60 * (m.sigma[j] + 1) + jssp.machines[machine].start_time - 24 * 60 * m.z[j] - (1-m.alpha[j, machine]) * H

    model.add_extra_time = pyo.Constraint(model.jobs, model.machines, rule=add_extra_time)

    return model


def solve_model(model: pyo.ConcreteModel):
    solver = pyo.SolverFactory("cplex")
    res = solver.solve(model)
    if res.solver.status != pyo.SolverStatus.ok:
        print("Check solver not ok...")
        raise Exception("Solver not ok")
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        print("Could not find optimal solution, probably infeasible...")
        log_infeasible_constraints(model, logger=logger, log_expression=True, log_variables=True)
        raise Exception("Infeasible solution")
    return model


def get_schedule(model: pyo.ConcreteModel):
    job_order = np.ones((len(model.machines), len(model.jobs)), dtype=int) * -2
    job_start_times = {j: model.t[j].value for j in model.jobs}
    jobs_assigned_on_machine = {m: 0 for m in model.machines}
    for j in sorted(job_start_times, key=job_start_times.get):
        found = False
        for m in model.machines:
            if model.alpha[j, m].value == 1 and not found:
                job_order[m, jobs_assigned_on_machine[m]] = j
                found = True
                jobs_assigned_on_machine[m] += 1
            elif found and model.alpha[j, m].value == 1:
                print("Error: Multiple machines assigned to job")
    return job_order

def validate_schedule(model: pyo.ConcreteModel, schedule: schedule_type) -> bool:
    for machine, jobs in schedule.items():
        for job_idx, start_time, end_time in jobs:
            if job_idx in [-1, -2]:
                continue
            if model.t[job_idx].value != start_time:
                print(f"Error: Start time does not match for job {job_idx}, machine {machine}, expected {model.t[job_idx].value}, got {start_time}")
                return False
    return True

if __name__ == "__main__":
    jssp = JobShopProblem.from_data(parse_data("examples/data_v1.xlsx"))
    print("Data parsed...")
    print("Generating model...")
    model = generate_model(jssp)
    print("Model generated...")
    print("Solving model...")
    solve_model(model)
    print("Model solved...")
    job_order = get_schedule(model)
    sc = jssp.make_schedule_from_parallel(job_order)
    print("Schedule generated...")
    model.pprint()
    print("Validating schedule...")
    valid = validate_schedule(model, sc)
    print(f"Is schedule valid {valid}")
    print("Tardiness: ", jssp.classical_tardiness(sc))
    print("Linear tardiness: ", model.objective())
    jssp.visualize_schedule(sc)
