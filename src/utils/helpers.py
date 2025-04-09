import os
import numpy as np
import random
from datetime import datetime

def save_figure(fig, name, output_dir="results"):
    """
    Saves a matplotlib figure to the specified output directory with a timestamp.

    The filename includes the provided name and a timestamp to avoid overwriting 
    previous figures. If the output directory doesn't exist, it is created.

    Args:
        fig: The matplotlib figure object to save.
        name: Base name for the saved figure.
        output_dir: Directory where the figure will be stored (default is "optimization_results").
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{name}_{timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")

def create_problem_instance(n_resources=8, n_tasks=12, seed=None):
    """
    Creates a reproducible instance of a resource allocation problem.

    This function generates:
    - A cost matrix of shape (n_resources x n_tasks), where each value represents
      the cost of assigning a resource to a task.
    - Resource capacity values for each resource, indicating the maximum amount
      they can handle.
    - Task requirement values for each task, specifying how much allocation is needed.

    If a random seed is provided, the outputs will be reproducible.

    Args:
        n_resources: Number of available resources (default is 8).
        n_tasks: Number of tasks that need resource allocation (default is 12).
        seed: Optional seed for reproducible randomness.

    Returns:
        A dictionary containing:
            - 'n_resources': Total number of resources.
            - 'n_tasks': Total number of tasks.
            - 'cost_matrix': A (n_resources x n_tasks) matrix of random costs.
            - 'resource_capacities': Array of capacity values for each resource.
            - 'task_requirements': Array of requirement values for each task.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    cost_matrix = np.random.rand(n_resources, n_tasks) * 100
    resource_capacities = np.random.randint(50, 100, size=n_resources)
    task_requirements = np.random.randint(10, 30, size=n_tasks)
    
    return {
        'n_resources': n_resources,
        'n_tasks': n_tasks,
        'cost_matrix': cost_matrix,
        'resource_capacities': resource_capacities,
        'task_requirements': task_requirements
    }
