from ..envs.world_model_env import WorldModelEnv

env = WorldModelEnv(record_video=False)
obs = env.reset()
print("Controller obs shape:", obs.shape)

for _ in range(50):
    a = env.action_space.sample()
    obs, r, done, info = env.step(a)
    if done:
        obs = env.reset()

env.close()
print("OK")
