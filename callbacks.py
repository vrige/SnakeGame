import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common import type_aliases
#from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import numpy as np
import gym
from typing import Union, Tuple, List

# the following callback saves the step reward in a pandas table at the end of each step, then it updates a history
# pandas table if the episode is completed. In this case, it computes the return of that episode. Finally, it creates
# a history csv file to be saved in the specified dir_path + "/Return", but it's possible to save also the spe-by-step
# table if the boolean step_bool is True.
# input: dir_path = path to the folder where the csv will be saved
#        step_bool(default=False) = save also step-by-step table
#        verbose (default=1) = 0 for no printing, 1 for printing returns and 2 for printing returns and rewards
# output: None
# Note: it works without a monitor, but future callbacks should implement it to share results
class EvaluationCallback_with_pandas(BaseCallback):
    def __init__(self, dir_path, step_bool=False, verbose=0):
        super(EvaluationCallback_with_pandas, self).__init__(verbose)

        self.step_bool = step_bool
        self.filepath = os.path.join(dir_path, f"Results")
        self.verbose = verbose

        # create a pandas fine-grained table where each row is a different step in an episode
        self.history_for_steps = pd.DataFrame({"episode": pd.Series(dtype=int),
                                               "n_steps_in_the_episode": pd.Series(dtype=int),
                                               "n_steps": pd.Series(dtype=int),
                                               "reward": pd.Series(dtype=int)})

        # create a pandas coarse grained table where each row is a different episode
        self.history = pd.DataFrame({"episode": pd.Series(dtype=int),
                                     "n_steps_in_the_episode": pd.Series(dtype=int),
                                     "n_steps": pd.Series(dtype=int),
                                     "return": pd.Series(dtype=int),
                                     "done?": pd.Series(dtype=bool)})

    def _on_training_start(self) -> None:

        # create a folder "Return" on the specified file_path
        if not os.path.exists(self.filepath):
            if self.verbose >= 1:
                print("creating the dir Return inside the path: " + self.filepath)
            os.makedirs(self.filepath)

        # Iterate over key/value pairs in dict and print them
        if self.verbose == 2:
            print("printing the avaiable local and global variables")
            if self.locals is not None:
                print("--------------------locals----------------------")
                for key, value in self.locals.items():
                    print(key, ' : ', value)
            if self.globals is not None:
                print("--------------------globals----------------------")
                for key, value in self.globals.items():
                    print(key, ' : ', value)
        self.episode = 1
        self.prev_number_of_step = 0
        pass

    def _on_step(self) -> bool:
        # if the episode is finished, then save in the table the results
        if self.locals["dones"][0] == True:
            df = pd.DataFrame({"episode": self.episode,
                               "n_steps_in_the_episode": self.locals["n_steps"] - self.prev_number_of_step,
                               "n_steps": self.locals["n_steps"],
                               "return": pd.Series.sum(self.history_for_steps.loc[self.history_for_steps["episode"] == self.episode]["reward"]),
                               "done?": False}, index=[self.episode])
            self.history = pd.concat([self.history, df])
            self.episode += 1
            self.prev_number_of_step = self.locals["n_steps"]
            if self.verbose >= 1:
                print(df)

        df = pd.DataFrame({"episode": self.episode,
                           "n_steps_in_the_episode": self.locals["n_steps"] - self.prev_number_of_step,
                           "n_steps": self.locals["n_steps"],
                           "reward": self.locals["rewards"]}, index=[self.n_calls])
        self.history_for_steps = pd.concat([self.history_for_steps, df])
        if self.verbose == 2:
            print(df)

        return True

    def _on_training_end(self) -> None:

        # save the table with all the rewards step-by-step in a csv file
        if self.step_bool:
            if self.verbose == 2:
                print("creating history and history_for_steps csv files in the path: " + self.filepath)
            self.history_for_steps.to_csv(os.path.join(self.filepath, "history_for_steps.csv"), header=True, index=False)

        # save the returns of each episode in a csv file
        if self.verbose == 2:
            print("creating history csv file in the path: " + self.filepath)
        self.history.to_csv(os.path.join(self.filepath, "history.csv"), header=True, index=False)
        pass

# wrapper for taking into account the episodes, the return of each episode and their lengths.
# Inputs:
# - env -> the environment to wrap
# - size -> the initial size of the step-by-step reward array
# - verbose -> (default=1) = 0 for no printing, 1 for printing returns and 2 for printing returns and rewards
class WrapperEpisodes(gym.Wrapper):
    def __init__(self, env: gym.Env, size: int = 250, verbose: int = 0):
        super(WrapperEpisodes, self).__init__(env)
        self.verbose = verbose
        self.episode_count = 0
        self.steps_count = 0
        self.episode_rewards = np.empty(size, dtype=float)
        self.episodes_rewards = []
        self.episode_return = 0
        self.episodes_returns = np.empty(0, dtype=float)
        self.episode_length = 0
        self.episodes_lengths = np.empty(0, dtype=float)
        self.episodes_steps = np.empty(0, dtype=float)

        # The following two fields are needed because the callback is not called exactly after
        # the step, but after the reset. So, the value of the end of the episode are saved in these
        # fields that aren't changed after reset method
        self.done_episode_length = self.episode_length
        self.done_episode_return = self.episode_return

    def reset(self, **kwargs):
        obs = super().reset()
        self.episode_rewards = np.empty(self.episode_rewards.size, dtype=float)
        self.episode_length = 0
        self.episode_return = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.episode_length += 1
        self.steps_count += 1

        # if the length of array is too short, then double its size
        if self.episode_length == self.episode_rewards.size:
            tmp = np.empty(self.episode_rewards.size, dtype=float)
            self.episode_rewards = np.concatenate((self.episode_rewards, tmp), axis=None)

        self.episode_rewards[self.episode_length] = reward
        self.episode_return += reward

        # if the episode is finished, then save the results in the appropriate numpy array
        if done:
            self.done_episode_return = self.episode_return
            self.done_episode_length = self.episode_length
            if self.verbose != 0:
                print('Episode: {}, len episode: {}, return episode: {}'.format(self.episode_count, self.episode_length, self.episode_return))
            self.episode_count += 1
            self.episodes_rewards.append(self.episode_rewards)
            self.episodes_returns = np.concatenate((self.episodes_returns, [self.episode_return]), axis=None)
            self.episodes_lengths = np.concatenate((self.episodes_lengths, [self.episode_length]), axis=None)
            self.episodes_steps = np.concatenate((self.episodes_steps, [self.steps_count]), axis=None)
            if self.verbose == 2:
                print("rewards: " + str(self.episodes_returns))
                print("lengths: " + str(self.episodes_lengths))

        return obs, reward, done, info

    # return the length of each episode as an array
    def get_episode_lengths(self):
        return self.episodes_lengths.astype(int)

    # return the length of the last episode
    def get_done_episode_length(self):
        return self.done_episode_length

    # return the returns of each episode
    def get_episodes_returns(self):
        return self.episodes_returns

    # return the return of the last episode
    def get_done_episode_return(self):
        return self.done_episode_return

    # return the number of steps
    def get_total_steps(self):
        return self.steps_count

    # return the total number of seen episodes
    def get_total_episodes(self):
        return self.episode_count

    # return the steps for each episode as an array
    def get_episodes_steps(self):
        return self.episodes_steps.astype(int)

    # set the mode for rendering
    def set_mode(self, mode='nothing'):
        super().render(mode)


# This callback works with the previous wrapper WrapperEpisodes.
# This function has two purposes:
# - save the results from the training and testing
# - evaluate with a certain frequency the model and if it is good enough save it
# Inputs:
# - model -> the model that we are evaluating
# - dir_path -> the path to the dir where the callback is going to save the results
# - eval_freq -> in steps, the frequency with which the model must be evaluated
# - n_eval_episodes -> the number of episodes to evaluate during the testing
# - render -> boolean for rendering the simulations
# - deterministic -> boolean for deciding the type of policy during the testing
# - step_bool -> boolean for saving the data step-by-step
# - eval_env -> FIX-ME
# - verbose -> int for the verbosity of the output
class evaluateCallback_withWrapper(BaseCallback):
    def __init__(self, model, dir_path, eval_freq=10000, n_eval_episodes: int = 10,
                 render=False, deterministic=False,
                 step_bool=False, eval_env=Union[gym.Env, VecEnv], verbose=0):
        super(evaluateCallback_withWrapper, self).__init__(verbose)
        self.model = model
        self.training_env = model.get_env()
        self.verbose = verbose
        self.step_bool = step_bool
        self.filepath = os.path.join(dir_path, f"Results")
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.render = render
        self.deterministic = deterministic

        self.callat = np.empty(0, dtype=int)
        self.episodes_n = np.empty(0, dtype=int)
        self.evaluations_results = np.empty(0, dtype=float)
        self.evaluations_length = np.empty(0, dtype=float)
        self.best_mean_reward = 0

    def _on_training_start(self) -> None:
        # create the dir to save the results in if it doesn't exist
        if not os.path.exists(self.filepath):
            if self.verbose >= 1:
                print("creating the dir Return inside the path: " + self.filepath)
            os.makedirs(self.filepath)
        pass

    def _on_step(self) -> bool:
        # if the episode is finished and the function is in verbose-mode, then
        # print the results of each step of the training
        if self.locals["dones"]:
            if self.verbose >= 1:
                print('Episode: {}, len episode: {}, return episode: {}'.format(
                    *self.training_env.env_method("get_total_episodes"),
                    *self.training_env.env_method("get_done_episode_length"),
                    *self.training_env.env_method("get_done_episode_return")))

        # with a certain frequency (based on the number of steps) check the model by testing it.
        # Save anyway the results and if it is better than the last saved model, save the model.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # call a custom function for testing
            episodes_return, episode_lengths = evaluate_policy_forCustomWrappers(self.model,
                                                                                 self.training_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True)
            mean_reward, std_reward = np.mean(episodes_return), np.std(episodes_return)
            #mean_length, std_length = np.mean(episode_lengths), np.std(episode_lengths)

            # save the return, length and episode of each episode
            # callat is the frequency at which the program decided to test the model
            self.callat = np.concatenate((self.callat, [self.n_calls] * self.n_eval_episodes), axis=None)
            self.episodes_n = np.concatenate((self.episodes_n, [range(1, self.n_eval_episodes + 1)]), axis=None)
            self.evaluations_results = np.concatenate((self.evaluations_results, episodes_return), axis=None)
            self.evaluations_length = np.concatenate((self.evaluations_length, episode_lengths), axis=None)

            if self.verbose >= 1:
                print("evaluating at step: " + str(self.n_calls) + ", ep mean reward: " +
                      str(mean_reward) + ", ep mean length: " + str(std_reward))

            # if the mean return is better than the previous one, save the new model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")

                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.filepath, str(self.n_calls)))
        return True
    def _on_training_end(self) -> None:
        if self.step_bool:
            if self.verbose == 2:
                print("creating history and history_for_steps csv files in the path: " + self.filepath)

            # fix-me: implementing fine-grained table

        if self.verbose == 2:
             print("creating history csv file in the path: " + self.filepath)

        # save the data from the training in a csv file
        tmp = self.training_env.env_method("get_total_episodes")
        n_epis = np.arange(1, int(*tmp) + 1)
        lengths = self.training_env.env_method("get_episode_lengths")
        steps = self.training_env.env_method("get_episodes_steps")
        returns = self.training_env.env_method("get_episodes_returns")
        my_array = np.transpose(np.array([n_epis, *lengths, *steps, *returns], dtype=object))
        df = pd.DataFrame(my_array, columns=['episode', 'n_steps_in_the_episode', 'n_steps', 'return'])
        df.to_csv(os.path.join(self.filepath, "history.csv"), header=True, index=False)

        # save the data from the testing in a csv file
        my_array = np.transpose(np.array([self.callat, self.episodes_n, self.evaluations_results,
                                          self.evaluations_length], dtype=object))
        df = pd.DataFrame(my_array, columns=['steps_in_training', 'episode', 'return', 'length'])
        df.to_csv(os.path.join(self.filepath, "testing.csv"), header=True, index=False)
        pass

# function for evaluating the model
# Inputs:
# - model -> the model that must be tested
# - env -> the environment for the testing
# - n_eval_episodes -> the number of episodes to evaluate during the testing
# - render -> boolean for rendering the simulations
# - deterministic -> boolean for deciding the type of policy during the testing
# - return_episode_rewards -> boolean for choosing the kind of return of the function
def evaluate_policy_forCustomWrappers(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    return_episode_rewards: bool = False
):
    episodes_return = []
    episode_lengths = []
    episode_counts = 0

    current_rewards = 0
    current_lengths = 0
    observations = env.reset()
    states = None

    # rending the simulation if it is selected
    if render:
        model.get_env().env_method("set_mode", mode='human')

    # iterate over a fix number of episodes
    while episode_counts < n_eval_episodes:

        # predict the next action and state
        actions, states = model.predict(observations, state=states, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        # if the episode is finished, save the results
        if dones:
            episodes_return.append(current_rewards)
            episode_lengths.append(current_lengths)
            episode_counts += 1
            current_rewards = 0
            current_lengths = 0

    # compute mean and std of returns of all the episodes
    mean_reward = np.mean(episodes_return)
    std_reward = np.std(episodes_return)

    # at the end of the simulations remove the rending option
    if render:
        model.get_env().env_method("set_mode", mode='console')

    if return_episode_rewards:
        return episodes_return, episode_lengths
    return mean_reward, std_reward
