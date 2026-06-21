import gym
from gym import spaces
import numpy as np

class TokamakVDEEnv(gym.Env):
    def __init__(self):
        super(TokamakVDEEnv, self).__init__()
        # State: [vertical position z, vertical velocity z_dot]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)
        # Action: Voltage control input u
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        
        self.gamma = 1.5   
        self.b = 2.0       
        self.dt = 0.001    
        self.max_steps = 200
        
    def reset(self):
        self.state = np.array([np.random.uniform(-0.1, 0.1), 0.0], dtype=np.float32)
        self.current_step = 0
        return self.state

    def step(self, action):
        z, z_dot = self.state
        u = action[0]
        
        # Physics update
        z_double_dot = (self.gamma**2) * z + self.b * u
        new_z_dot = z_dot + z_double_dot * self.dt
        new_z = z + new_z_dot * self.dt
        
        self.state = np.array([new_z, new_z_dot], dtype=np.float32)
        self.current_step += 1
        
        # Reward engineering
        reward = -(10.0 * (z**2) + 0.1 * (u**2))
        done = bool(abs(new_z) > 1.0 or self.current_step >= self.max_steps)
        if abs(new_z) > 1.0:
            reward -= 100.0 
            
        return self.state, reward, done, {}