from stable_baselines3 import PPO
import os
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env import SneakEnv
from stable_baselines3.common.env_util import make_vec_env
from time import time

# useful links: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/3_multiprocessing.ipynb#scrollTo=zlJSs7TmZ3jf
# https://github.com/hill-a/stable-baselines/issues/322#issuecomment-492202915

# create some folders to save the results in
models_dir = f"models/result"

for i in range(100):
	models_dir = f"models/result_" + str(i)
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
		break

processors = [2, 4, 8, 16, 32, 64, 128]
TIMESTEPS = 100000
time_sub = []
time_dummy = []
time_normal = 0

for i in processors:

	print("----------------------------------------- ", i, " ---------------------------------------------------------")
	if i == 2:
		# create a SubprocVecEnv custom env for the snake game
		env = SneakEnv(dim=100)
		env.reset()
		model = PPO('MlpPolicy', env, verbose=0)
		start = time()
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
		end = time()
		time_normal = round(end - start, 3)
		print("normal it needed ", time_normal)

	# create a SubprocVecEnv custom env for the snake game
	env_sub = make_vec_env(SneakEnv, n_envs=i, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'), env_kwargs={"dim": 100})
	env_sub.reset()

	# create a DummyVecEnv custom env for the snake game
	env_dummy = make_vec_env(SneakEnv, n_envs=i, vec_env_cls=DummyVecEnv, env_kwargs={"dim": 100})
	env_dummy.reset()

	model_sub = PPO('MlpPolicy', env_sub, verbose=0)
	model_dummy = PPO('MlpPolicy', env_dummy, verbose=0)

	start_sub = time()
	model_sub.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
	end_sub = time()
	print("sub with ", i, " processors, it needed ", round(end_sub - start_sub, 3))

	start_dummy = time()
	model_dummy.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
	end_dummy = time()
	print("dummy with ", i, " processors, it needed ", round(end_dummy - start_dummy, 3))

	time_sub.append(round(end_sub - start_sub, 3))
	time_dummy.append(round(end_dummy - start_dummy, 3))

print(processors)
print(time_sub)
print(time_dummy)

import pandas as pd
d = {'col1': time_dummy, 'col2': time_dummy}
print(pd.DataFrame(data=d, index=processors))

# timesteps:     100000
# processors:	   1        2, 		 4, 	 8,	     16, 	   32, 	    64, 	128
# SubprocVecEnv: 		 66.959   61.619   56.94   61.733    52.162   50.613   93.945
# DummyVecEnv:  	     58.587   55.884   51.323  55.221    46.052   46.041   90.214
# no vector:    61.385

# timesteps:     200000
# processors:	   1        2,      4, 	    8,        16,     32,       64, 	128
# SubprocVecEnv: 	129.043, 109.243, 103.451,  93.853, 104.552, 100.272,  99.601
# DummyVecEnv:  	104.208, 101.861,  96.188,  83.833, 109.159,  94.108,  92.497
# no vector:    131.969

# timesteps:     400000
# processors:	   1        2, 		 4, 	 8, 	  16, 	  32, 	    64, 	128
# SubprocVecEnv:		 236.044  203.962  189.736  251.075  186.624  207.332  213.203
# DummyVecEnv:			 207.035  174.376  158.807  188.2    189.302  183.297  211.013
# no vector:    257.019