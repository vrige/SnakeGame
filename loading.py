from stable_baselines3 import PPO
from env import SnekEnv
from stable_baselines3.common.env_checker import check_env

env = SnekEnv(rending=True)
#check_env(env)
env.reset()

# load a model
models_dir = f"models/noRend1/Results"
model_path = f"{models_dir}/100000.zip"
model = PPO.load(model_path, env=env)

episodes = 15

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()  # it doesn't have a specific render function
        #print(rewards)

env.close()
