import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class FakeSERLObsWrapper(gym.ObservationWrapper):
    """
    Simplified observation wrapper for fake environment.
    Assumes state is already a flat array and just passes it through.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        # For fake env, we ignore proprio_keys and just use the flat state
        self.proprio_keys = proprio_keys or []
        print(f"FakeSERLObsWrapper: ignoring proprio_keys {self.proprio_keys}")
        
        # Create observation space that matches what the classifier expects
        self.observation_space = gym.spaces.Dict(
            {
                "state": self.env.observation_space["state"],  # Keep state as-is (1,9) array
                # Include all image keys at top level for classifier
                **{k: v for k, v in self.env.observation_space.spaces.items() 
                   if k not in ["state", "images"]},  # wrist_1, wrist_2
            }
        )

    def observation(self, obs):
        result = {
            "state": obs["state"],  # Keep chunked state 
        }
        
        # Add image keys at top level, ensuring correct 4D shape
        for key in ["wrist_1", "wrist_2"]:
            if key in obs:
                img = obs[key]
                # Remove extra dimensions: (batch, 1, 1, h, w, c) -> (batch, h, w, c)
                if len(img.shape) == 6:  # (batch, 1, 1, h, w, c)
                    img = img[:, 0, 0]   # (batch, h, w, c)
                elif len(img.shape) == 5:  # (batch, 1, h, w, c)
                    img = img[:, 0]      # (batch, h, w, c)
                result[key] = img
                
        return result

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info