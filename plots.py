from pandas import read_csv
import numpy as np
import os
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

# This function is a multipurpose function that:
# - compute the Confidence intervals for the lengths and returns of the testing phase taking the data from the
#   specified path
# - save the dataframe with extra information in a csv folder
# - create two plots for the means and CI of both returns and lengths
# - save the two plots in one pdf
# This function is coupled with the following function draw_graph.
# Inputs:
# - path is the path of the csv file with testing data
# - alpha is the level of error
# - size is in relation with mode parameter and they are both used to decide how to plot the output
# - mode is in relation with size parameter and they are both used to decide how to plot the output
# - separate is true, then two graphs are drawn in two separate pdf. Otherwise, hey are drawn  in the same pdf.
# - color to be used in the graphs

def plot_results(path, alpha=0.05, separate=False, size=2, mode='none', color='#003f5c'):

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
    n_sim = df["n_simul"].max()
    # Confidence levels
    Conf = 1 - alpha / 2
    conf_level = (1 - alpha) * 100

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
        draw_graph(df, root, "Returns and length in testing", n_sim, conf_level, size, mode, color)
    else:
        # draw two graphs in two separate pdf
        draw_graph(df, root, "return", n_sim, conf_level, size, mode, color, separate=True)
        draw_graph(df, root, "length", n_sim, conf_level, size, mode, color, separate=True)

# function coupled with the previous function plot_results and the structure of the csv from the training.
# The purpose of this function is to save the graphs with the means and the CI in the root_path.
# The original function may be modified using the parameters mode and size.
# Inputs:
# - df is the dataframe realized with the function plot_results
# - root_path is the path of the dir where the csv file for testing is
# - name to be used in the graphs
# - n_simulation the number of simulations
# - size is in relation with mode parameter and they are both used to decide how to plot the output
# - mode is in relation with size parameter and they are both used to decide how to plot the output
# - color to be used in the graphs
# - separate is true, then two graphs are drawn in two separate pdf. Otherwise, they are drawn in the same pdf.
def draw_graph(df, root_path, name, n_simulation, conf_level, size=2,
               mode='none', color='#003f5c', separate=False):

    # clean previous plot
    plt.clf()
    # steps for the x_axis
    x = df['n_steps'].to_numpy()
    # set the scale format by getting the frequency and then finding the largest exponent
    # in base 10 such that that power is lower than x
    check = x
    if len(x) > 1:
        check = math.gcd(x[0], x[1])
    min_scale = scale_check(check)

    if not separate:

        # drawing graphs
        fig, (ax1, ax2) = plt.subplots(2)
        fig.tight_layout(pad=3.5)

        # means and confidence intervals graph for return
        y = df['return_mean'].to_numpy()
        ci = df['return_CI'].to_numpy()
        x_, lower = switch_graph_method(x, y - ci, size, mode)
        x_, upper = switch_graph_method(x, y + ci, size, mode)
        x_, y_ = switch_graph_method(x, y, size, mode)
        ax1.plot(x_, y_)
        ax1.fill_between(x_, lower, upper, color=color, alpha=.2)
        ax1.set_xlabel('steps')
        ax1.set_ylabel('return')
        ax1.title.set_text('Return mean of {} simulations with {}% CI with mode {}'
                           .format(n_simulation, conf_level, mode))
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(min_scale, min_scale))

        # means and confidence intervals graph for length
        y = df['length_mean'].to_numpy()
        ci = df['length_CI'].to_numpy()
        x_, lower = switch_graph_method(x, y - ci, size, mode)
        x_, upper = switch_graph_method(x, y + ci, size, mode)
        x_, y_ = switch_graph_method(x, y, size, mode)
        ax2.plot(x_, y_)
        ax2.fill_between(x_, lower, upper, color=color, alpha=.2)
        ax2.set_xlabel('steps')
        ax2.set_ylabel('length')
        ax2.title.set_text('Length mean of {} simulations with {}% CI with mode {}'
                           .format(n_simulation, conf_level, mode))
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(min_scale, min_scale))

        # save graphs in the same pdf
        fig.savefig(os.path.join(root_path, "returns_and_length_CI.pdf"), bbox_inches='tight')

    else:

        # plot only one graph at the time
        y = df[name + "_mean"].to_numpy()
        ci = df[name + "_CI"].to_numpy()
        x_, lower = switch_graph_method(x, y - ci, size, mode)
        x_, upper = switch_graph_method(x, y + ci, size, mode)
        x_, y_ = switch_graph_method(x, y, size, mode)
        plt.plot(x_, y_)
        plt.fill_between(x_, lower, upper, color=color, alpha=.2)
        plt.xlabel('steps')
        plt.ylabel(name)
        plt.title('{} mean of {} simulations with {}% CI with mode {}'.
                  format(name, n_simulation, conf_level, mode))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(min_scale, min_scale))

        # save graphs in the same pdf file
        plt.savefig(os.path.join(root_path, name + "_CI.pdf"), bbox_inches='tight')


# this function will find the largest exponent in base 10 such that that power is lower than x
def scale_check(x):
    for i in range(0, 20):
        a = x / pow(10, i)
        if a < 1:
            return i - 1


# This function compute a smoothing curve using the cubic interpolation.
# the function takes input some points (x, y) and it computes a smoothing function on each step
# from the first value of x to the last one
def smoothing_cubid(x, y):
    x_smo = np.arange(x[0], x[-1], 1)
    y_smo = interp1d(x, y, kind='cubic')
    return x_smo, y_smo(x_smo)


# This function returns the simple moving average of the array y
def simple_moving_average(y):
    cumsum = np.cumsum(y)
    return np.divide(cumsum, np.arange(1, len(cumsum)+1, 1))


# This function returns the simple moving average of the array y, but it uses also a window N.
# If the size of the window is lower or equal to 1, greater than the size of the array y or it isn't an integer,
# then an exception is raised.
# Attention: the first N have a lower size window, so to them is applied the simple_moving_average without window
def simple_moving_average_with_window(y, N):
    if len(y) <= N or N <= 1 or not isinstance(N, int):
        raise Exception("the window for the simple_moving_average_with_window function must be an integer"
                        "greater than 1 and lower than the data size")
    sma = simple_moving_average(y[:N])
    cumsum = np.cumsum(y)
    return np.concatenate((sma, (cumsum[N:] - cumsum[:-N]) / float(N)))


# This function returns the weighted moving average of the array y
def weighted_moving_average(y):
    w = np.arange(1, len(y)+1, 1)
    cumsum = np.cumsum(np.multiply(w, y))
    return np.divide(cumsum, np.cumsum(w))


# This function returns the weighted moving average of the array y, but it uses also a window N.
# If the size of the window is lower or equal to 1, greater than the size of the array y or it isn't an integer,
# then an exception is raised.
# Attention: the first N have a lower size window, so to them is applied the weighted_moving_average without window
def weighted_moving_average_with_window(y, N):
    if len(y) <= N or N <= 1 or not isinstance(N, int):
        raise Exception("the window for the weighted_moving_average_with_window function must be an integer"
                        "greater than 1 and lower than the data size")
    w = np.arange(1, N+1, 1)
    wma = weighted_moving_average(y[:(N-1)])
    matrix = np.zeros((len(y)-N + 1, N))
    for i in range(len(y)-N + 1):
        matrix[i, :] = np.multiply(w, y[i:i+N])
    return np.concatenate((wma, np.divide(np.sum(matrix, axis=1), float(np.sum(w)))))


# This function returns the exponential moving average of the array y, but it uses also a window N to set
# the initial value using the simple moving average.
def exponential_moving_average(y, N):
    if len(y) <= N or N < 1 or not isinstance(N, int):
        raise Exception("the initial size of sma for the exponential_moving_average function must be an integer"
                        "greater than 0 and lower than the data size")
    sma = simple_moving_average(y[:N])
    k = 2 / (N + 1)
    ema = np.concatenate((sma, np.zeros(len(y) - N)))
    for i in range(N, len(y), 1):
        ema[i] = y[i] * k + ema[i-1] * (1-k)
        k = 2 / (i + 1)
    return ema


# Function for modifying the points (x, y). The chosen modifications must be specified in the mode
# parameter and the available modifications are:
# - "none"  = no modifications
# - "smo"   = Cubic interpolation
# - "sma"   = simple moving average
# - "sma-w" = simple moving average with window of size N
# - "wma"   = weighted moving average
# - "wma-w" = weighted moving average with window of size N
# - "ema"   = exponential moving average with initial size of the sma equal to N
# - "show"  = it prints a plot with all using all the previous techniques
# the output is always the pairs (x, y)
# Attention: the parameter N is just ignored when the chosen mode is "smo","sma" or "wma"
# Furthermore, when mode is different from the available ones, then an exception is raised.
def switch_graph_method(x, y, N, mode="none"):
    if mode == "none":
        return x, y
    elif mode == "smo":
        return smoothing_cubid(x, y)
    elif mode == "sma":
        return x, simple_moving_average(y)
    elif mode == "sma-w":
        return x, simple_moving_average_with_window(y, N)
    elif mode == "wma":
        return x, weighted_moving_average(y)
    elif mode == "wma-w":
        return x, weighted_moving_average_with_window(y, N)
    elif mode == "ema":
        return x, exponential_moving_average(y, N)
    elif mode == "show":
        show(x, y, N)
        return x, y
    else:
        raise Exception("The chosen mode is not available. The available modes are: smo, sma, sma-w, wma, wma-w, ema")

# this function will plot the functions obtained with the methods available in the switch_graph_method function
# in the same plot. It will ask for keyboard pressing at the end of the function.
def show(x,y,size):
    mode_list = ["none", "smo", "sma", "sma-w", "wma", "wma-w", "ema"]
    plt.clf()
    for i in mode_list:
        x_, y_ = switch_graph_method(x, y, size, i)
        plt.plot(x_, y_, label=i)
        plt.xlabel('steps')
        plt.ylabel(i)
    plt.legend(loc="lower right")
    plt.show()
    input("Have a look at the plot then press Enter to continue...")

# many things to be fixed:
# 5- creating a folder for each training with specific parameters
# 6- somewhere if we train a new model with some parameters already used, we should cumulate the data all in
#    one csv, so we can again use all these methods
