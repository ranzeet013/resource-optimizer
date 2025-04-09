# Resource Allocation Optimizer

AI-powered resource allocation system that uses a hybrid approach combining genetic algorithms and neural networks to optimize resource allocation under constraints.

## Features

- Hybrid optimization approach (Genetic Algorithm + Neural Network)
- Constraint-aware resource allocation
- Real-time visualization of solutions
- Automatic saving of optimization results and visualizations
- Configurable problem parameters

## Project Structure

```
resource-optimizer/
├── src/
│   ├── utils/
│   │   └── helpers.py         # Utility functions
│   ├── models/
│   │   └── problem.py         # Problem definition
│   ├── visualization/
│   │   └── visualizer.py      # Visualization functions
│   ├── optimization/
│   │   └── hybrid_optimizer.py # Hybrid AI optimizer
│   └── main.py                # Main execution script
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resource-optimizer.git
cd resource-optimizer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the optimizer:
```bash
python src/main.py
```

The script will:
1. Create a problem instance with configurable parameters
2. Run the hybrid optimization process
3. Save visualizations to the `optimization_results` directory
4. Display the final solution and performance metrics

## Configuration

You can modify the following parameters in `src/main.py`:
- Number of resources
- Number of tasks
- Population size
- Number of generations
- Random seed for reproducibility

## Output

The optimizer generates:
- Resource allocation matrices
- Resource utilization plots
- Task allocation vs requirements plots
- Cost contribution matrices
- Fitness convergence history

All visualizations are automatically saved to the `optimization_results` directory with timestamps.

## License

MIT License 