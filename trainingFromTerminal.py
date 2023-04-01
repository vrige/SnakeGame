import argparse
import os
from stable_baselines3 import PPO
from env import SnekEnv
from callbacks import EvaluationCallback_with_pandas, WrapperEpisodes, evaluateCallback_withWrapper
from plots import plot_results
# type in the terminal
# python .\trainingFromTerminal.py --h

# arguments for the parser for the environment
def env_argument_parser(parser):
        parser.add_argument(
                "--Trend",  # name on the CLI - drop the `--` for positional/required parameters
                type=bool,
                default=False,
                help="Rending during training")
        return parser

# arguments for the parser for the wrapper
def wrapper_argument_parser(parser):
        parser.add_argument(
                "--Wverb",  # name on the CLI - drop the `--` for positional/required parameters
                type=int,
                default=0,
                help="The verbosity of the wrapper")
        parser.add_argument(
                "--Isize",  # name on the CLI - drop the `--` for positional/required parameters
                type=int,
                default=250,
                help="The initial size for the wrapper")
        return parser

# arguments for the parser for the callback
def callback_argument_parser(parser):
        parser.add_argument(
                "--Neval",  # name on the CLI - drop the `--` for positional/required parameters
                type=int,
                default=35,
                help="The number of simulation done by the callback for testing")
        parser.add_argument(
                "--Feval",  # name on the CLI - drop the `--` for positional/required parameters
                type=int,
                default=10000,
                help="The frequency for testing in the callback")
        parser.add_argument(
                "--Cverb",  # name on the CLI - drop the `--` for positional/required parameters
                type=int,
                default=0,
                help="The verbosity of the callback")
        parser.add_argument(
                "--RendT",  # name on the CLI - drop the `--` for positional/required parameters
                type=bool,
                default=False,
                help="Rending during testing in the training")

        return parser

# arguments for the parser for the ppo algorithm
def ppo_argument_parser(parser):
    parser.add_argument(
        "--v",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=0,
        help="The verbosity of the model")
    parser.add_argument(
        "--p",
        choices=['MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy',
                 'CnnPolicy', 'CnnLstmPolicy', 'CnnLnLstmPolicy'],
        default='MlpPolicy',
        help="The policy of the algorithm ppo")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00003,
        help="The learning rate of the algorithm ppo")
    parser.add_argument(
        "--st",
        type=int,
        default=2048,
        help="The number of steps of the algorithm ppo")
    parser.add_argument(
        "--bs",
        type=int,
        default=64,
        help="The batch size of the algorithm ppo")
    parser.add_argument(
        "--ne",
        type=int,
        default=10,
        help="The number of epochs of the algorithm ppo")
    parser.add_argument(
        "--g",
        type=float,
        default=0.99,
        help="The gamma parameter for the algorithm ppo")
    parser.add_argument(
        "--gl",
        type=float,
        default=0.95,
        help="The gae_lambda parameter for the algorithm ppo")
    return parser

# arguments for the parser for the dqn algorithm
def dqn_argument_parser(parser):
    parser.add_argument(
        "--v",  # name on the CLI - drop the `--` for positional/required parameters
        type=int,
        default=0,
        help="The verbosity of the model")
    parser.add_argument(
        "--p",
        choices=['MlpPolicy', 'MlpLstmPolicy', 'MlpLnLstmPolicy',
                 'CnnPolicy', 'CnnLstmPolicy', 'CnnLnLstmPolicy'],
        default='MlpLstmPolicy',
        help="The policy of the algorithm dqn")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="The learning rate of the algorithm dqn")
    parser.add_argument(
        "--bus",
        type=int,
        default=1000000,
        help="The buffer_size parameter for the algorithm dqn")
    parser.add_argument(
        "--ls",
        type=int,
        default=50000,
        help="The learning_starts parameter for the algorithm dqn")
    parser.add_argument(
        "--bs",
        type=int,
        default=32,
        help="The batch_size parameter for the algorithm dqn")
    parser.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="The tau parameter for the algorithm dqn")
    parser.add_argument(
        "--g",
        type=float,
        default=0.99,
        help="The gamma parameter for the algorithm dqn")
    parser.add_argument(
        "--tf",
        type=int,
        default=4,
        help="The train frequency parameter for the algorithm dqn")
    parser.add_argument(
        "--gs",
        type=int,
        default=1,
        help="The gradient_steps parameter for the algorithm dqn")
    return parser

# create a dir in a path specified inside the dict variable args
# the found path should not contain any keys contained in env_args
def create_dir(args, env_args):

    for key, value in args.items():

        if key in env_args:
            continue

        args["path_results"] += str(key) + "=" + str(value) + "_"

    args["path_results"] = args["path_results"][:-1]

    if not os.path.exists(args["path_results"]):
        print("Directory {} created".format(args["path_results"]))
        os.makedirs(args["path_results"])

    return args["path_results"]

# create a text file with name "parameters.txt" inside the specified path
# and fill the text file with the content of the dict args
def print_info_file(args, path):
    filepath = os.path.join(path, "paramters.txt")
    with open(filepath, "w") as f:

        for keys, values in args.items():
            f.write("{}: {} \n".format(keys, values))

# train a model using one line from the terminal and save the results
def main():

        logdir = f"logs"

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # the point is that there is a way to directly choose the parser that you like without the necessary
        # to create all the possible parsers (just create a first dict and check the Algor field and then add
        # the arguments for the specific algorithm). However, the problem is on the user-side in which
        # if she/he does --h, she/he cannot see all the parameters and it may get very confusing.

        # final parser that will be extended using subparsers
        # the trick is to keep this parser empty and pass the shared arguments of the parent parser
        # using another parser (parser_parent)
        parser = argparse.ArgumentParser(
            prog='PROG',
            epilog="See '<command> --help' to read about a specific sub-command."
        )
        parser_parent = argparse.ArgumentParser(add_help=False)

        # adding arguments for envorinment, wrapper and callbacks
        parser_parent = callback_argument_parser(wrapper_argument_parser(env_argument_parser(parser_parent)))

        # extract the list of arguments not related to the algorithm
        list_elem_env = list(vars(parser_parent.parse_known_args()[0]))

        # create subparsers from the main parser but using as parent parser_parent
        # specifying the dest we are asking the user to select an algorithm
        subparsers = parser.add_subparsers(dest='algorithm', help='Sub-commands')

        # ppo parser
        ppo_parser = subparsers.add_parser('ppo', help='parser for ppo algorithm', parents=[parser_parent])
        ppo_parser = ppo_argument_parser(ppo_parser)

        # dqn parser
        dqn_parser = subparsers.add_parser('dqn', help='parser for dqn algorithm', parents=[parser_parent])
        dqn_parser = dqn_argument_parser(dqn_parser)

        # extract the main parser
        args_ns = parser.parse_args()

        # create pathfile
        # notice that the folder will have a path similar to this: "result/name_algorithm/arguments"
        # the mentioned arguments will be the ones related to only the algorithm
        args_ns.path_results = "results/"
        args_ns.path_results += str(args_ns.algorithm) + "/"

        # get the dict from the extracted arguments of the parser
        args = vars(args_ns)

        # extend the list of the env arguments
        list_elem_env.extend(['algorithm', 'path_results', 'v']) # v is the verbosity of the algorithm

        # Create path dir if not exists
        pathfile = create_dir(args, list_elem_env)

        # create a text file with all the arguments
        print_info_file(args, pathfile)

        # create the custom env for the snake game
        env = SnekEnv(rending=args["Trend"])
        env.reset()

        # create a wrapper to keep track of the episodes
        wrapper = WrapperEpisodes(env, args["Wverb"], args["Isize"])
        wrapper.reset()

        if args_ns.algorithm == "ppo":

            # create a model using a specific algorithm
            model = PPO(args["p"], wrapper, args["v"], tensorboard_log=logdir)

            # create a callback that works with the wrapper
            evalcallback = evaluateCallback_withWrapper(model, dir_path=pathfile, n_eval_episodes=args["Neval"],
                                                        eval_freq=args["Feval"], verbose=args["Cverb"],
                                                        render=args["RendT"])
            callbacks = [evalcallback]

            # train the model and track it on tensorboard
            model.learn(total_timesteps=args["st"], reset_num_timesteps=False, tb_log_name=f"PPO", callback=callbacks)

            # plot the results with the CI and save the testing in a csv file
            #plot_results(os.path.join(pathfile, "testing.csv"))
            #plot_results(os.path.join(pathfile, "testing.csv"), separate=True)

            pass

        elif args_ns.algorithm == "dqn":
            print("FIXME")
            pass

if __name__ == '__main__': main()