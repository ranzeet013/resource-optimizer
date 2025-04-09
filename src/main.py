"""
- Hybrid optimization using evolutionary and neural techniques
- Constraint-aware allocation with adaptive behavior
- Automated visualization of the optimization process (saved to disk)
"""

import os
from utils.helpers import create_problem_instance
from models.problem import ResourceAllocationProblem
from optimization.hybrid_optimizer import HybridAIOptimizer
from visualization.visualizer import visualize_solution

def main():
    print("=== Advanced AI Resource Allocation Optimizer ===")
    print("Creating problem instance...")
    
    problem_data = create_problem_instance(n_resources=8, n_tasks=12, seed=42)
    problem = ResourceAllocationProblem(problem_data)
    
    print("\nProblem parameters:")
    print(f"- Resources: {problem.n_resources}")
    print(f"- Tasks: {problem.n_tasks}")
    print(f"- Resource capacities: {problem.resource_capacities}")
    print(f"- Task requirements: {problem.task_requirements}")
    
    print("\nStarting optimization...")
    optimizer = HybridAIOptimizer(problem, pop_size=100, n_generations=30)
    best_solution, best_fitness = optimizer.run_optimization()
    
    print("\n=== Optimization Complete ===")
    print(f"Best solution found with fitness: {best_fitness:.2f}")
    print("\nAllocation matrix (resources x tasks):")
    print(best_solution.reshape(problem.n_resources, problem.n_tasks).round(2))
    
    visualize_solution(problem, best_solution)
    
    print("\nAll visualizations saved to:", os.path.abspath("optimization_results"))

if __name__ == "__main__":
    main()
