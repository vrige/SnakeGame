from stable_baselines3 import PPO
import os
from env import SnekEnv
from callbacks import EvaluationCallback_with_pandas, WrapperStatistics, evaluateCallback_withWrapper
from plots import plot_results

# create some folders to save the results in
models_dir = f"models/noRend1"
logdir = f"logs/noRend1"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)
if not os.path.exists(logdir):
	os.makedirs(logdir)

# create the custom env for the snake game
env = SnekEnv(rending=False)
env.reset()

# create a wrapper to keep track of the episodes
wrapper = WrapperStatistics(env, 250, verbose=0)
wrapper.reset()

# create a model using a specific algorithm
model = PPO('MlpPolicy', wrapper, verbose=1, tensorboard_log=logdir)

# number of time steps
TIMESTEPS = 200000

# create a callback that works with the wrapper
evalcallback = evaluateCallback_withWrapper(model, dir_path=models_dir, eval_freq=10000, verbose=0)
callbacks = [evalcallback]

# train the model and track it on tensorboard
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=callbacks)

# plot the results with the CI and save the testing in a csv file
plot_results(os.path.join(models_dir, "Results/testing.csv"))
plot_results(os.path.join(models_dir, "Results/testing.csv"), separate=True)

