import matplotlib.pyplot as plt
import numpy as np
from ..utils.helpers import save_figure

def visualize_solution(problem, solution, generation=None, output_dir="optimization_results"):
    """
    Visualizes the allocation results of the optimization process.

    Displays:
    - Resource allocation matrix
    - Resource utilization vs capacity
    - Task allocation vs requirements
    - Cost contribution matrix

    Saves the figure to the specified output directory.
    
    Args:
        problem: ResourceAllocationProblem instance containing problem parameters.
        solution: Flattened array representing the allocation matrix.
        generation: Optional generation number for labeling.
        output_dir: Directory where the figure will be saved.
    """
    allocation = solution.reshape(problem.n_resources, problem.n_tasks)
    
    fig = plt.figure(figsize=(12, 8))
    
    if generation is not None:
        plt.suptitle(f"Generation {generation}", y=1.02)
    
    plt.subplot(2, 2, 1)
    plt.imshow(allocation, cmap='viridis', aspect='auto')
    plt.title("Resource Allocation Matrix")
    plt.xlabel("Tasks")
    plt.ylabel("Resources")
    plt.colorbar(label="Allocation Amount")
    
    plt.subplot(2, 2, 2)
    resource_usage = np.sum(allocation, axis=1)
    plt.bar(range(problem.n_resources), resource_usage, label='Usage')
    plt.bar(range(problem.n_resources), problem.resource_capacities, 
           alpha=0.3, label='Capacity')
    plt.title("Resource Utilization")
    plt.xlabel("Resources")
    plt.ylabel("Amount")
    plt.legend()
    
    plt.subplot(2, 2, 3)
    task_allocation = np.sum(allocation, axis=0)
    plt.bar(range(problem.n_tasks), task_allocation, label='Allocated')
    plt.bar(range(problem.n_tasks), problem.task_requirements, 
           alpha=0.3, label='Required')
    plt.title("Task Allocation vs Requirements")
    plt.xlabel("Tasks")
    plt.ylabel("Amount")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    cost_contribution = allocation * problem.cost_matrix
    plt.imshow(cost_contribution, cmap='hot', aspect='auto')
    plt.title("Cost Contribution Matrix")
    plt.xlabel("Tasks")
    plt.ylabel("Resources")
    plt.colorbar(label="Cost")
    
    plt.tight_layout()
    
    if generation is not None:
        save_figure(fig, f"generation_{generation}", output_dir)
    else:
        save_figure(fig, "final_solution", output_dir)
    
    plt.show()
    plt.close(fig)

def plot_fitness_history(fitness_history, output_dir="optimization_results"):
    """
    Plots the history of best fitness values over generations.

    Args:
        fitness_history: List of best fitness values per generation.
        output_dir: Directory where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(fitness_history)+1), fitness_history, 'b-')
    ax.set_title("Best Fitness Over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Cost + Penalties)")
    ax.grid(True)
    save_figure(fig, "fitness_history", output_dir)
    plt.show()
    plt.close(fig)
