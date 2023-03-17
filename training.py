from stable_baselines3 import PPO
import os
from env import SnekEnv
from callbacks import EvaluationCallback_with_pandas, WrapperStatistics, evaluateCallback_withWrapper

models_dir = f"models/noRend1"
logdir = f"logs/noRend1"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SnekEnv(rending=False)
env.reset()

wrapper = WrapperStatistics(env, 250, verbose=0)
wrapper.reset()

model = PPO('MlpPolicy', wrapper, verbose=1, tensorboard_log=logdir)
# for retraining a model
#model_path = f"models/noRend1/150000.zip"
#model = PPO.load(model_path, env=env, tensorboard_log=logdir)

TIMESTEPS = 100000

evalcallback = evaluateCallback_withWrapper(model, dir_path=models_dir, eval_freq=10000, verbose=0)
callbacks=[evalcallback]

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO", callback=callbacks)
