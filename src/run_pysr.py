import numpy as np
from pysr import PySRRegressor

def discover_equations():
    print("Loading trajectory data...")
    X_data = np.load("X_data.npy")
    y_data = np.load("y_data.npy")
    
    sr_model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "-"],
        unary_operators=[],
        select_k_features=2,
        model_selection="best"
    )
    
    print("Initiating Physics-Informed Symbolic Regression...")
    sr_model.fit(X_data, y_data)
    
    print("\nDiscovery Complete. Pareto Front:")
    print(sr_model.equations_)
    print("\nOptimal Symbolic Law:")
    print(sr_model.sympy())

if __name__ == "__main__":
    discover_equations()