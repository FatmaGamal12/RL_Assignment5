from envs.atari_env import AtariEnv

env = AtariEnv(env_id="BreakoutNoFrameskip-v4")

obs = env.reset()
print("Initial obs shape:", obs.shape)

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if done:
        break

env.close()
print("Video saved.")
