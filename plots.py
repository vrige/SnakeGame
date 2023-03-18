from pandas import read_csv
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

# This function is a multipurpose function that:
# - compute the Confidence intervals for the lengths and returns of the testing phase taking the data from the
#   specified path
# - save the dataframe with extra information in a csv folder
# - create two plots for the means and CI of both returns and lengths
# - save the two plots in one pdf
# This function is coupled with the following function draw_graph.
# Inputs:
# - path of the csv file with testing data
# - the level of error alpha
# - separate is true, then two graphs are drawn in two separate pdf. Otherwise, hey are drawn  in the same pdf.
# - color to be used in the graphs
def plot_results(path, alpha = 0.05, separate = False, color = '#003f5c'):

    # extract the root from the path
    root = os.path.split(path)[0]

    # read the csv from the training phase
    df = read_csv(path)

    # create a dataframe by grouping by the number of steps
    x_steps = df["steps_in_training"].unique()
    a = df.groupby(['steps_in_training'])["episode"].max()
    df = df.drop(columns="episode")
    df = df.groupby(['steps_in_training']).mean().join(df.groupby(['steps_in_training']).std(), lsuffix='_mean',
                                                       rsuffix='_std')
    df["n_simul"] = a
    df['return_std'].replace(np.nan, 0)
    df['length_std'].replace(np.nan, 0)
    df['n_steps'] = x_steps

    # number of simulations
    n_sim = df["n_simul"].iloc[1]
    # Confidence levels
    Conf = 1 - alpha / 2
    conf_level = 1 - alpha

    # independent simulations -> gaussian if the sample size is greater or equal to 30 (because the central limit theorem
    # only holds true when the sample size is “sufficiently large"), so it's better to use a t-score (t-student)
    # in case it is lower than 30.
    # notice that the degree of freedom of the t-student are n-1 where n is the sample size
    # and when we compute, for instance, the 95% confidence interval, then it's 2.5% from left and 2.5 from right
    # ASSUMPTION: to simplify the code, we can assume that the number of simulations for each call is the same.
    # So, we don't need to check row by row which formula to use
    if n_sim >= 30:
        df['return_CI'] = df.apply(lambda x: stats.norm.ppf(Conf) * x['return_std'] / np.sqrt(x['n_simul']), axis=1)
        df['length_CI'] = df.apply(lambda x: stats.norm.ppf(Conf) * x['length_std'] / np.sqrt(x['n_simul']), axis=1)
    else:
        df['return_CI'] = df.apply(lambda x: stats.t(df=n_sim - 1).ppf(Conf) * x['return_std'] / np.sqrt(x['n_simul']),
                                   axis=1)
        df['length_CI'] = df.apply(lambda x: stats.t(df=n_sim - 1).ppf(Conf) * x['length_std'] / np.sqrt(x['n_simul']),
                                   axis=1)
    # save dataframe in csv
    df.to_csv(os.path.join(root, "testing_CI.csv"), header=True, index=False)

    if not separate:
        # draw two graphs in the same pdf
        draw_graph(df, root, "Returns and length in testing", color)
    else:
        # draw two graphs in two separate pdf
        draw_graph(df, root, "return", color, separate = True)
        draw_graph(df, root, "length", color, separate = True)

# function coupled with the previous function plot_results and the structure of the csv from the training.
# The purpose of this function is to save the graphs with the means and the CI in the root_path.
# Inputs:
# - df is the dataframe realized with the function plot_results
# - root_path is the path of the dir where the csv file for testing is
# - name to be used in the graphs
# - color to be used in the graphs
# - separate is true, then two graphs are drawn in two separate pdf. Otherwise, they are drawn in the same pdf.
def draw_graph(df, root_path, name, color = '#003f5c', separate = False):

    # clean previous plot
    plt.clf()
    # steps for the x_axis
    x = df['n_steps']

    if not separate:

        # drawing graphs
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(name)
        fig.tight_layout()

        # means and confidence intervals graph for return
        y = df['return_mean'].to_numpy()
        ci = df['return_CI'].to_numpy()
        lower = (y - ci).tolist()
        upper = (y + ci).tolist()
        ax1.plot(x, y)
        ax1.fill_between(x, lower, upper, color=color, alpha=.2)
        ax1.set_xlabel('steps')
        ax1.set_ylabel('return')

        # means and confidence intervals graph for length
        y = df['length_mean'].to_numpy()
        ci = df['length_CI'].to_numpy()
        lower = (y - ci).tolist()
        upper = (y + ci).tolist()
        ax2.plot(x, y)
        ax2.fill_between(x, lower, upper, color=color, alpha=.2)
        ax2.set_xlabel('steps')
        ax2.set_ylabel('length')

        # save graphs in the same pdf
        fig.savefig(os.path.join(root_path, "returns_and_length_CI.pdf"), bbox_inches='tight')

    else:

        # plot only one graph at the time
        y = df[name + "_mean"].to_numpy()
        ci = df[name + "_CI"].to_numpy()
        lower = (y - ci).tolist()
        upper = (y + ci).tolist()
        plt.plot(x, y)
        plt.fill_between(x, lower, upper, color=color, alpha=.2)
        plt.xlabel('steps')
        plt.ylabel(name)

        # save graphs in the same pdf
        plt.savefig(os.path.join(root_path, name + "_CI.pdf"), bbox_inches='tight')

