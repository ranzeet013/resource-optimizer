import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from ..visualization.visualizer import visualize_solution, plot_fitness_history

class HybridAIOptimizer:
    """
    HybridAIOptimizer performs optimization using a hybrid approach combining a genetic algorithm
    with a neural network-based surrogate model for faster fitness evaluation.
    
    Attributes:
        problem: An object with a callable `evaluate(individual)` method defining the optimization problem.
        pop_size: Number of individuals in the population.
        n_generations: Number of generations to run the genetic algorithm.
        surrogate_model: Neural network model used as a fitness surrogate.
        scaler: StandardScaler to normalize input features for the surrogate model.
        toolbox: DEAP toolbox containing registered evolutionary operations.
    """
    def __init__(self, problem, pop_size=100, n_generations=50):
        """
        Initializes the optimizer with problem definition and genetic algorithm setup.

        Args:
            problem: The optimization problem object with an 'evaluate' method.
            pop_size: Size of the population (default: 100).
            n_generations: Number of generations to evolve (default: 50).
        """
        self.problem = problem
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.surrogate_model = None
        self.scaler = StandardScaler()
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, 
                              n=self.problem.n_resources * self.problem.n_tasks)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
    def train_surrogate_model(self, X, y):
        """
        Trains a neural network surrogate model to approximate the fitness function.

        Args:
            X: Training input features (population individuals).
            y: Corresponding true fitness values.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.surrogate_model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
        self.surrogate_model.fit(X_scaled, y)
        
    def surrogate_evaluate(self, individual):
        """
        Evaluates the fitness of an individual using the trained surrogate model.

        Args:
            individual: A single individual (solution vector) from the population.

        Returns:
            Predicted fitness value from the surrogate model.
        """
        X_scaled = self.scaler.transform([individual])
        return self.surrogate_model.predict(X_scaled)[0]
    
    def run_optimization(self, output_dir="optimization_results"):
        """
        Runs the hybrid optimization process combining genetic algorithm and surrogate model.

        Args:
            output_dir: Directory to store visualizations and results (default: "optimization_results").

        Returns:
            best_ind: Best solution (individual) found after all generations.
            best_fitness: True fitness value of the best solution.
        """
        pop = self.toolbox.population(n=self.pop_size)
        
        print("Evaluating initial population...")
        fitnesses = list(map(self.problem.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,)
            
        initial_best = tools.selBest(pop, 1)[0]
        visualize_solution(self.problem, initial_best, generation=0, output_dir=output_dir)
            
        X = np.array(pop)
        y = np.array(fitnesses)
        self.train_surrogate_model(X, y)
        
        fitness_history = [min(fitnesses)]
        
        for gen in range(self.n_generations):
            print(f"\nGeneration {gen+1}/{self.n_generations}")
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.surrogate_evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)
                
            pop[:] = offspring
            
            current_best = tools.selBest(pop, 1)[0]
            best_fitness = self.problem.evaluate(current_best)
            fitness_history.append(best_fitness)
            print(f"Current best fitness: {best_fitness:.2f}")
            
            if gen % 5 == 0 or gen == self.n_generations - 1:
                print("Re-training surrogate model...")
                sample_indices = np.random.choice(len(pop), size=min(20, len(pop)), replace=False)
                sample_X = np.array([pop[i] for i in sample_indices])
                sample_y = np.array([self.problem.evaluate(ind) for ind in sample_X])
                
                X = np.vstack([X, sample_X])
                y = np.concatenate([y, sample_y])
                self.train_surrogate_model(X, y)
                
                visualize_solution(self.problem, current_best, generation=gen+1, output_dir=output_dir)
                
        plot_fitness_history(fitness_history, output_dir=output_dir)
        
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = self.problem.evaluate(best_ind)
        
        return best_ind, best_fitness
