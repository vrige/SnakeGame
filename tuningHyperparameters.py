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

    reward_system = {
        "apple": 10000,
        "nothing": 0,
        "die": -1000,
        "dist": True,
        "dist_f": lambda x, y: np.linalg.norm(x - y),
        "tot_rew": lambda x, y, z: (250 - reward_system["dist_f"](x, y) + z) / 100
                    if (reward_system["dist"]) else (250 + z) / 100
    }

    # create the custom env for the snake game
    eval_env = Monitor(SneakEnv(reward_system=reward_system, rending=False))
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
[I 2023-04-09 11:28:33,007] A new study created in memory with name: no-name-c6523846-8956-4f60-b6fd-7a4699003e36
[I 2023-04-09 11:29:33,362] Trial 0 finished with value: -20.323461333333334 and parameters: {'lr': 0.09345756527828554, 'exponent_n_steps': 14, 'exponent_batch_size': 7, 'n_epoches': 10, 'gamma': 0.742441555512156, 'gae_lambda': 0.7749889988132174, 'ent_coef': 1.0654872824091608e-06, 'vf_coef': 1.0563003600739571e-07, 'max_grad_norm': 0.3111076661892704, 'normalize_advantage': True}. Best is trial 0 with value: -20.323461333333334.
Number of finished trials:  2
Best trial:
  Value:  -4.136033333333334
  Params: 
    lr: 0.00021791348291227945
    exponent_n_steps: 18
    exponent_batch_size: 5
    n_epoches: 20
    gamma: 0.9902749463214242
    gae_lambda: 0.5476525588418972
    ent_coef: 2.3999203924972125e-07
    vf_coef: 0.000607547698237474
    max_grad_norm: 2.865597112860597
    normalize_advantage: True
  User attrs:
[I 2023-04-09 11:53:04,917] Trial 1 finished with value: -4.136033333333334 and parameters: {'lr': 0.00021791348291227945, 'exponent_n_steps': 18, 'exponent_batch_size': 5, 'n_epoches': 20, 'gamma': 0.9902749463214242, 'gae_lambda': 0.5476525588418972, 'ent_coef': 2.3999203924972125e-07, 'vf_coef': 0.000607547698237474, 'max_grad_norm': 2.865597112860597, 'normalize_advantage': True}. Best is trial 1 with value: -4.136033333333334.

Process finished with exit code 0

'''