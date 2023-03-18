from stable_baselines3 import PPO
from env import SnekEnv
from stable_baselines3.common.env_checker import check_env

# create a virtual env for snake game
env = SnekEnv(rending=True)
env.reset()

# check if the env is working
#check_env(env)

# load a model
models_dir = f"models/noRend1/Results"
model_path = f"{models_dir}/100000.zip"
model = PPO.load(model_path, env=env)

# number of simulations
episodes = 15

# simulate the model
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()
        #print(rewards)

env.close()
