import numpy as np

class ResourceAllocationProblem:
    """
    Defines a resource allocation optimization problem.

    Attributes:
        n_resources (int): Number of resources available.
        n_tasks (int): Number of tasks to be completed.
        cost_matrix (np.ndarray): Matrix representing the cost of assigning resources to tasks.
        resource_capacities (np.ndarray): Maximum capacity of each resource.
        task_requirements (np.ndarray): Required amount of work for each task.
    """

    def __init__(self, problem_data):
        """
        Initialize the problem with data.

        Args:
            problem_data (dict): Dictionary containing:
                - n_resources (int)
                - n_tasks (int)
                - cost_matrix (2D array)
                - resource_capacities (1D array)
                - task_requirements (1D array)
        """
        self.n_resources = problem_data['n_resources']
        self.n_tasks = problem_data['n_tasks']
        self.cost_matrix = problem_data['cost_matrix']
        self.resource_capacities = problem_data['resource_capacities']
        self.task_requirements = problem_data['task_requirements']
        
    def evaluate(self, allocation):
        """
        Evaluate a solution based on cost and constraint violations.

        Args:
            allocation (np.ndarray): A flat array representing allocation of resources to tasks.

        Returns:
            float: Total cost including penalties for constraint violations.
        """
        total_cost = 0
        penalty = 0
        allocation = allocation.reshape(self.n_resources, self.n_tasks)
        total_cost = np.sum(allocation * self.cost_matrix)
        resource_usage = np.sum(allocation, axis=1)
        capacity_violation = np.maximum(resource_usage - self.resource_capacities, 0)
        penalty += np.sum(capacity_violation) * 1000
        task_allocation = np.sum(allocation, axis=0)
        requirement_violation = np.abs(task_allocation - self.task_requirements)
        penalty += np.sum(requirement_violation) * 500
        return total_cost + penalty
