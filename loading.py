from stable_baselines3 import PPO
from env import SneakEnv
from stable_baselines3.common.env_checker import check_env

# create a virtual env for snake game
env = SneakEnv(rending=True, dim=500, time_speed=0.1)
env.reset()

# check if the env is working
#check_env(env)

# load a model
models_dir = f"results/ppo/learning_rate=3e-05_batch_size=64_n_epochs=10_gamma=0.99_gae_lambda=0.95_normalize_advantage=True_ent_coef=0_vf_coef=0.5_max_grad_norm=0.5_seed=None"
model_path = f"{models_dir}/100000.zip"
model = PPO.load(model_path, env=env)


# number of simulations
episodes = 10

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
