#credits: https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py

from typing import Any
from typing import Dict

import gym
from stable_baselines3 import PPO
from env import SneakEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from callbacks import WrapperEpisodes, evaluateCallback_withWrapper
import optuna
import os
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import joblib

N_EVALUATIONS = 5
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 5
N_TRIALS = 150
N_STARTUP_TRIALS = 5

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy"
}

"""Sampler for PPO hyperparameters."""
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:

    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 10, 19)
    b_size = 2 ** trial.suggest_int("exponent_batch_size", 2, 8)
    n_epoches = trial.suggest_int("n_epoches", 1, 20)
    gamma = trial.suggest_float("gamma", 0.5, 1, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.00000001, 0.7, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [True, False])

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": b_size,
        "n_epochs": n_epoches,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "normalize_advantage": normalize_advantage
    }

"""Callback used for evaluating and reporting a trial."""
class TrialEvalCallback(EvalCallback):

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))

    # create the custom env for the snake game
    eval_env = Monitor(SneakEnv(rending=False))
    eval_env.reset()

    # create a wrapper to keep track of the episodes
    #wrapper = WrapperEpisodes(eval_env, 250, verbose=0)
    #wrapper.reset()

    # Create the RL model.
    model = PPO(env=eval_env, **kwargs, verbose=0)

    # create a callback that works with the wrapper
    #evalcallback = evaluateCallback_withWrapper(model, dir_path=models_dir, n_eval_episodes=35,
    #                                            eval_freq=10000, verbose=0, render=False)
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    callbacks = [eval_callback]

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=callbacks)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


# Set pytorch num threads to 1 for faster training.
torch.set_num_threads(1)

sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used.
pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
try:
    study.optimize(objective, n_trials=N_TRIALS, timeout=600)
except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print("    {}: {}".format(key, value))


# create some folders to save the results in
models_dir = f"optuna_results"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# to save study to savepath + "xgb_optuna_study_batch.pkl"
savepath = models_dir
joblib.dump(study, f"{savepath}/snake_ppo_tuning.pkl")   # save study

# to load it:
#jl = joblib.load(f"{savepath}xgb_optuna_study_batch.pkl")

#print(jl.best_trial.params)
'''
[I 2023-04-11 10:04:34,764] A new study created in memory with name: no-name-4ae53d4e-1844-4437-8082-c1c80b40836b
[I 2023-04-11 10:05:00,908] Trial 0 finished with value: 8.106089599999999 and parameters: {'lr': 0.00706981696463819, 'exponent_n_steps': 11, 'exponent_batch_size': 8, 'n_epoches': 6, 'gamma': 0.9370831424072001, 'gae_lambda': 0.5680893963980804, 'ent_coef': 1.992013431937773e-06, 'vf_coef': 0.0012540365129698676, 'max_grad_norm': 0.7165258640211681, 'normalize_advantage': True}. Best is trial 0 with value: 8.106089599999999.
[I 2023-04-11 10:09:16,217] Trial 1 finished with value: -23.5619328 and parameters: {'lr': 0.03212322574358495, 'exponent_n_steps': 12, 'exponent_batch_size': 2, 'n_epoches': 7, 'gamma': 0.5859895869462459, 'gae_lambda': 0.758823410332787, 'ent_coef': 0.001992086793082053, 'vf_coef': 0.010219090336029694, 'max_grad_norm': 1.0903228690304987, 'normalize_advantage': False}. Best is trial 0 with value: 8.106089599999999.
Number of finished trials:  3
Best trial:
  Value:  8.106089599999999
  Params: 
    lr: 0.00706981696463819
    exponent_n_steps: 11
    exponent_batch_size: 8
    n_epoches: 6
    gamma: 0.9370831424072001
    gae_lambda: 0.5680893963980804
    ent_coef: 1.992013431937773e-06
    vf_coef: 0.0012540365129698676
    max_grad_norm: 0.7165258640211681
    normalize_advantage: True
  User attrs:
[I 2023-04-11 10:24:53,336] Trial 2 finished with value: -6.017462399999999 and parameters: {'lr': 4.443615044433846e-05, 'exponent_n_steps': 14, 'exponent_batch_size': 2, 'n_epoches': 17, 'gamma': 0.6462480549832594, 'gae_lambda': 0.6130253096945992, 'ent_coef': 0.05815745026936087, 'vf_coef': 1.1749742540586791e-08, 'max_grad_norm': 1.0196489879672814, 'normalize_advantage': True}. Best is trial 0 with value: 8.106089599999999.


'''