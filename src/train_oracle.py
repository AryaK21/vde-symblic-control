import numpy as np
from stable_baselines3 import PPO
from env import TokamakVDEEnv

def generate_trajectory_data():
    env = TokamakVDEEnv()
    print("Initializing environment and training PPO oracle...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.003)
    model.learn(total_timesteps=40000)
    
    print("Training complete. Harvesting operational trajectories...")
    X_data, y_data = [], []
    obs = env.reset()
    
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        X_data.append(obs.copy())
        y_data.append(action[0])
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()
            
    # Save the data for PySR to use later
    np.save("X_data.npy", np.array(X_data))
    np.save("y_data.npy", np.array(y_data))
    print("Data successfully saved to disk.")
    
    return model

if __name__ == "__main__":
    generate_trajectory_data()