import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import TokamakVDEEnv

def run_benchmarks(model):
    print("--- Running Inference Latency Benchmarks ---")
    sample_state = np.array([[0.1, 0.05]], dtype=np.float32)
    
    # 1. Neural Network Benchmark
    start_time = time.time()
    for _ in range(10000):
        _ = model.predict(sample_state, deterministic=True)
    nn_time = (time.time() - start_time) / 10000 * 1e6 
    
    # 2. Native Algebraic Benchmark
    z, z_dot = 0.1, 0.05
    start_time = time.time()
    for _ in range(10000):
        u = -0.0915 * z + 0.2161 * z_dot + 0.2374
    sr_time = (time.time() - start_time) / 10000 * 1e6 
    
    print(f"PPO Neural Network: {nn_time:.3f} µs")
    print(f"PiSR Algebraic Law: {sr_time:.3f} µs")
    print(f"Speedup Factor: {nn_time / sr_time:.1f}x faster")

def generate_publication_graphs():
    z, z_dot, dt, gamma, b = 0.1, 0.0, 0.001, 1.5, 2.0
    z_history, z_dot_history, u_history, time_steps = [], [], [], []

    for t in range(500): 
        time_steps.append(t * dt)
        z_history.append(z)
        z_dot_history.append(z_dot)
        
        # PiSR Discovered Control Law
        u = -0.0915 * z + 0.2161 * z_dot + 0.2374
        u_history.append(u)
        
        z_double_dot = (gamma**2) * z + b * u
        z_dot += z_double_dot * dt
        z += z_dot * dt

    # Matplotlib styling for academic papers
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

    # Plot 1: Stabilization Profile
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Vertical Position $z$ (m)', color='tab:blue')
    ax1.plot(time_steps, z_history, color='tab:blue')
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Actuator Voltage $u$ (V)', color='tab:red')  
    ax2.plot(time_steps, u_history, color='tab:red', linestyle='--')
    
    fig.tight_layout()  
    plt.savefig("vde_stabilization_plot.pdf", format='pdf', bbox_inches='tight')

    # Plot 2: Phase-Space
    plt.figure(figsize=(6, 6))
    plt.plot(z_history, z_dot_history, color='tab:purple')
    plt.scatter(z_history[0], z_dot_history[0], color='green', zorder=5)
    plt.scatter(z_history[-1], z_dot_history[-1], color='red', zorder=5)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Vertical Position $z$ (m)')
    plt.ylabel('Vertical Velocity $\dot{z}$ (m/s)')
    plt.savefig("phase_space_plot.pdf", format='pdf', bbox_inches='tight')
    print("Graphs saved successfully.")

if __name__ == "__main__":
    # You would load the trained model here to benchmark it
    # For now, we just run the graphs
    generate_publication_graphs()