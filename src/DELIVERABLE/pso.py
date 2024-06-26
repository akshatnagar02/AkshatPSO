import numpy as np
import random
from collections import deque, defaultdict
from enum import Enum
from job_shop_problem import JobShopProblem, ScheduleError, ObjectiveFunction
import time


class PSO:
    def __init__(self, jssp, num_particles=100, max_iter=200, inertia_max=0.9, inertia_min=0.4, phi1=2, phi2=1.5, objective_function=ObjectiveFunction.CUSTOM_OBJECTIVE):
        self.jssp = jssp
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia_max = inertia_max
        self.inertia_min = inertia_min
        self.phi1 = phi1
        self.phi2 = phi2
        self.objective_function = objective_function
        self.particles = []
        self.velocities = []
        self.p_best = []
        self.g_best = None
        self.best_solution = None
        self.best_score = float('inf')
        self.max_reset = 0  
        self.init_particles()

    def topological_sort(self):
        ''' 
        Performs a topological sort on the job dependencies to determine a valid sequence of jobs.
        
        Returns:
            sorted_jobs (list): A list of job indices in a topologically sorted order.
        '''
        in_degree = {i: 0 for i in range(len(self.jssp.jobs))}
        graph = defaultdict(list)

        for job_idx, job in enumerate(self.jssp.jobs):
            for dep in job.dependencies:
                graph[dep].append(job_idx)
                in_degree[job_idx] += 1

        queue = deque([job_idx for job_idx in in_degree if in_degree[job_idx] == 0])
        sorted_jobs = []

        while queue:
            job_idx = queue.popleft()
            sorted_jobs.append(job_idx)
            for neighbor in graph[job_idx]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_jobs) != len(self.jssp.jobs):
            raise ScheduleError("Circular dependency detected")

        return sorted_jobs

    def grouped_prod(self):
        ''' 
        Groups jobs together by production order number.
        
        Returns:
            grouped_jobs (defaultdict): A dictionary where each key is a production order number
                                        and each value is a list of job indices that correspond to
                                        that production order number.
        '''
        grouped_jobs = defaultdict(list)
        for idx, job in enumerate(self.jssp.jobs):
            grouped_jobs[job.production_order_nr].append(idx)
        return grouped_jobs

    def init_particles(self):
        '''
        Initializes entire swarm of particles including their positions and velocities for the first iteration. Jobs are considered to be grouped by production order.
        Also stores initial global and personal best positions.
        '''
        grouped_jobs = self.grouped_prod()
        for _ in range(self.num_particles):
            position = self.init_position(grouped_jobs)
            velocity = np.zeros(len(position))
            self.particles.append(position)
            self.velocities.append(velocity)
            self.p_best.append(position)
            if self.g_best is None or self.evaluate(position) < self.evaluate(self.g_best):
                self.g_best = position

    def init_position(self, grouped_jobs):
        '''
        Initializes feasible initial job sequences and machine assignments for each particle. Takes grouped_jobs as input and performs topological sort to respect job dependencies. Grouped jobs are then randomly assigned selected available machines using random.choice which is uniform random distribution function. 

        Arg:
            grouped_jobs (defaultdict): A dictionary where each key is a production order number and each value is a 
                                        list of job indices that correspond to that production order number.

        Returns:
            position (list): A list of tuples where each tuple contains the initial job and machine index.
        '''
        sorted_jobs = self.topological_sort()
        position = []
        for order_nr, jobs in grouped_jobs.items():
            for job in jobs:
                available_machines = list(self.jssp.jobs[job].available_machines.keys())
                machine = random.choice(available_machines)
                position.append((job, machine))
        return position

    def evaluate(self, position):
        '''
        Evaluates fitness of each particle. Generates a schedule based on job order and machine assignments used to calculate objective functions score.
        
        Args: 
            position (list): A list of tuples where each tuple contains the initial job and machine index.

        Returns:
            fitness (float): The fitness value of the given position with respect to custom objective function.
        '''
        job_orders = [[] for _ in range(len(self.jssp.machines))]
        for job, machine in position:
            job_orders[int(machine)].append(job)
        schedule = self.jssp.make_schedule_from_parallel_with_stock(job_orders)

        if self.objective_function == ObjectiveFunction.CUSTOM_OBJECTIVE:
            return self.jssp.custom_objective(schedule)
        elif self.objective_function == ObjectiveFunction.MAKESPAN:
            return self.jssp.makespan(schedule)
        elif self.objective_function == ObjectiveFunction.CLASSICAL_TARDINESS:
            return self.jssp.classical_tardiness(schedule)
        else:
            raise ValueError("Invalid objective function")

    def update_velocity(self, i, iter):
        ''' 
        First updates inertia weight, then updates velocity for each particle. r1 and r2 remain as uniformly distributed random arrays.

        Args:
            i (int): Index of the particle whose velocity is being updated.
            iter (int): Current iteration number, used to calculate the inertia weight.
        '''
        inertia_weight_update = self.inertia_max - ((self.inertia_max - self.inertia_min) / self.max_iter) * iter
        for j in range(len(self.velocities[i])):
            r1, r2 = random.random(), random.random()
            cognitive = self.phi1 * r1 * (self.p_best[i][j][1] - self.particles[i][j][1])
            social = self.phi2 * r2 * (self.g_best[j][1] - self.particles[i][j][1])
            self.velocities[i][j] = inertia_weight_update * self.velocities[i][j] + cognitive + social

    def update_position(self, i):
        ''' 
        Updates position of each particle using respective formula. Calls validate_position to see if new position respects job dependencies and machine availability.

        Args:
            i (int): Index of the particle whose position is being updated.
        '''
        for j in range(len(self.particles[i])):
            self.particles[i][j] = (self.particles[i][j][0], self.particles[i][j][1] + self.velocities[i][j])
            self.particles[i][j] = self.validate_position(self.particles[i][j])

    def validate_position(self, position):
        """
        Ensures valid job sequence. Checks if the machine assignment for a job is feasible by comparing it against the list of available machines for that job. If the machine assignment is not feasible it selects a valid machine from the available machines list. Job and Machine assignments are first unpacked, then the list of machines which jobs can be assigned to is retrieved. Machine assignment is then rounded as they need to be whole numbers to generate a schedule. If the machine assignment is not feasible, the function randomly selects a machine from the list of available machines. 

        Args:
            position (tuple): A tuple containing a job index and a machine assignment.

        Returns:
            (tuple): A tuple containing the job index and a validated machine assignment.
        """
        job, machine_assignment = position
        available_machines = list(self.jssp.jobs[job].available_machines.keys())
        machine_assignment = int(round(machine_assignment))
        if machine_assignment not in available_machines:
            machine_assignment = random.choice(available_machines)
        return (job, machine_assignment)
    
    def iterate(self, time_limit=300):
        '''
        Main loop of PSO algorithm. Updates the velocity and position of each particle, evaluates their fitness, and updates their personal and global best positions. 
        Time limit and convergence check functions are defined here, making the algorithm stop the iteration process if completion time exceeds 5 minutes, and checks if the best score has not improved significantly over a number of iterations it will reinitialize the particles.
        
        Args:
            time_limit (int): The maximum time (in seconds) allowed for the iterations. Default is 300 seconds (5 minutes).

        Returns:
            best_solution (list of tuples): Best solution found by the PSO algorithm within iteration process.
            elapsed_time (float): Total time taken for the iterations.
        '''
        start_time = time.time()
        for iter in range(self.max_iter):
            if time.time() - start_time > time_limit:
                print("Time limit reached, stopping...")
                break

            for i in range(self.num_particles):
                self.update_velocity(i, iter)
                self.update_position(i)

                current_fitness = self.evaluate(self.particles[i])
                if current_fitness < self.evaluate(self.p_best[i]):
                    self.p_best[i] = self.particles[i]
                if current_fitness < self.evaluate(self.g_best):
                    self.g_best = self.particles[i]

            self.g_best = self.local_search(self.g_best)

            best_solution_score = self.evaluate(self.g_best)
            if best_solution_score < self.best_score:
                self.best_score = best_solution_score
                self.best_solution = self.g_best

            if abs(best_solution_score - self.best_score) < 1e-6:
                self.max_reset += 1
            else:
                self.max_reset = 0

            if self.max_reset == 50:
                self.init_particles()
                self.max_reset = 0

        end_time = time.time()
        return self.best_solution, end_time - start_time

    def local_search(self, position):
        """
        Performs a local search to find a better job order by swapping jobs assigned to the same machine. 

        Args:
            position (list): Current best position to improve upon.

        Returns:
            best_position (list): Best position found during the local search.
        """
        best_position = position
        best_score = self.evaluate(position)
        max_local = 0

        for _ in range(20):
            machine = np.random.randint(len(self.jssp.machines))
            num_jobs = len([pos for pos in position if pos[1] == machine])
            if num_jobs < 2:
                continue

            job1_idx, job2_idx = np.random.choice(num_jobs, 2, replace=False)
            new_position = position[:]
            new_position[job1_idx], new_position[job2_idx] = new_position[job2_idx], new_position[job1_idx]

            new_score = self.evaluate(new_position)
            if new_score < best_score:
                best_position = new_position
                best_score = new_score
                max_local = 0
            else:
                max_local += 1 
            if max_local >= 5:
                break
        return best_position

    def downtime_check(self, schedule):
        """
        Downtime check ensures that the end time of the previous job on the same machine is the start time of the next job. Applies this adjust to the schedule where unnecessary downtimes occur.
       
        Args:
            schedule (dict): The current schedule with potential gaps, where the keys are machine IDs
                             and the values are lists of tasks. Each task is a tuple (job_id, start_time, end_time).

        Returns:
            dict: The adjusted schedule with no gaps between tasks for each machine.
        """
        for machine_id, jobs in schedule.items():
            current_end_time = self.jssp.machines[machine_id].start_time
            for idx in range(len(jobs)):
                job_id, start_time, end_time = jobs[idx]
                if start_time < current_end_time:
                    start_time = current_end_time
                duration = end_time - start_time
                end_time = start_time + duration
                jobs[idx] = (job_id, start_time, end_time)
                current_end_time = end_time
            schedule[machine_id] = jobs
        return schedule
    
    def get_best_schedule(self):
        """
        Creates the optimal schedule from the best solution found by the PSO algorithm. For each machine, the global best solution is used to determine the sequence of jobs, and the function ensures there are no downtimes between tasks on each machine.

        Returns:
            dict: The adjusted schedule with no downtime between tasks for each machine.
        """
        job_orders = [[] for _ in range(len(self.jssp.machines))]
        for job, machine in self.best_solution:
            job_orders[machine].append(job)
        schedule = self.jssp.make_schedule_from_parallel_with_stock(job_orders)
        return self.downtime_check(schedule)
