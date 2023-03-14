import os
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import numpy as np
import gym

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

        # create a pandas fine-grained table where each row is a different step in an
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
        if self.step_bool:
            if self.verbose == 2:
                print("creating history and history_for_steps csv files in the path: " + self.filepath)
            self.history.to_csv(os.path.join(self.filepath, "history.csv"), header=True, index=False)
            self.history_for_steps.to_csv(os.path.join(self.filepath, "history_for_steps.csv"), header=True, index=False)
        else:
            if self.verbose == 2:
                print("creating history csv file in the path: " + self.filepath)
            self.history.to_csv(os.path.join(self.filepath, "history.csv"), header=True, index=False)
        pass


class WrapperStatistics(gym.Wrapper):
    def __init__(self, env: gym.Env, size: int = 250, verbose: int = 0):
        super(WrapperStatistics, self).__init__(env)
        self.verbose = verbose
        self.episode_count = 0
        self.steps_count = 0
        self.episode_rewards = np.empty(size, dtype=float)
        self.episodes_rewards = []
        self.episode_return = 0
        self.episodes_returns = np.empty(0, dtype=float)
        self.episode_length = 0
        self.episodes_lengths = np.empty(0, dtype=float)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.episode_rewards = np.empty(self.episode_rewards.size, dtype=float)
        self.episode_length = 0
        self.episode_return = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.episode_length += 1
        self.steps_count += 1

        if self.episode_length == self.episode_rewards.size:
            tmp = np.empty(self.episode_rewards.size, dtype=float)
            self.episode_rewards = np.concatenate((self.episode_rewards, tmp), axis=None)

        self.episode_rewards[self.episode_length] = reward
        self.episode_return += reward

        if done:
            if self.verbose != 0:
                print('Episode: {}, len episode: {}, return episode: {}'.format(self.episode_count, self.episode_length, self.episode_return))
            self.episode_count += 1
            self.episodes_rewards.append(self.episode_rewards)
            self.episodes_returns = np.concatenate((self.episodes_returns, [self.episode_return]), axis=None)
            self.episodes_lengths = np.concatenate((self.episodes_lengths, [self.episode_length]), axis=None)
            if self.verbose == 2:
                print("rewards: " + str(self.episodes_returns))
                print("lengths: " + str(self.episodes_lengths))

        return obs, reward, done, info

    def get_episode_lengths(self):
        return self.episodes_lengths

    def get_episode_rewards(self):
        return self.episodes_returns

    def get_total_steps(self):
        return self.steps_count

    def get_total_episodes(self):
        return self.episode_count

class example(BaseCallback):
    def __init__(self, verbose=0):
        super(example, self).__init__(verbose)
        self.env1 = self.training_env

    def _on_training_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        print("sono nella callback ")
        print("printing the avaiable local and global variables")
        if self.locals is not None:
            print("--------------------locals----------------------")
            for key, value in self.locals.items():
                print(key, ' : ', value)
        print("sono sempre nella callback")
        return True