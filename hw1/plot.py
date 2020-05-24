import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def average_smoothing(s, window_size):
    """
    Smooths a series of scalars to its moving window average.
    Inspired by: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    Args:
        s: a series of scalars
        window_size: the size of the moving window

    Returns:
        a np.array of the same length of series containing the averages
    """
    if window_size < 2:
        return s
    s = np.r_[s[window_size-1:0:-1], s, s[-2:-window_size-1:-1]]
    window = np.ones(window_size) / window_size
    smooth_s = np.convolve(s, window, mode='valid')
    if window_size % 2 == 0:
        smooth_s = smooth_s[(window_size//2-1):-(window_size//2)]
    else:
        smooth_s = smooth_s[(window_size//2):-(window_size//2)]
    return smooth_s


def plot_group(prefix, logdir, imagedir, individual=False, smoothing_ws=-1):
    """
    Plots a group of runs. A test accuracy over iteration plot containing all
    runs on the group is generated and saved.

    Args:
        prefix: name prefix shared by all runs of the group
        logdir: directory containing run logs
        imagedir: directory where images will be saved
        individual: if True, additional learning curves are generated for each
            run on the group
        smoothing_ws: window size for moving window average smoothing, set to
            -1 to disable smoothing
    """
    # get paths to run logfiles
    group_prefix = os.path.join(logdir, "{}*".format(prefix))
    logfiles = glob.glob(group_prefix)

    # gather data from logs
    labels = []
    iterations = []
    test_accuracies = []
    if individual:
        test_losses = []
        train_losses = []
    for logfile in logfiles:
        with open(logfile, "rb") as f:
            logs = pickle.load(f)
        labels.append(logs['label'])
        iterations.append(logs['iteration'])
        test_accuracies.append(logs['test_accuracy'])
        if individual:
            test_losses.append(logs['test_loss'])
            train_losses.append(logs['train_loss'])

    # smooth data if needed
    if smoothing_ws > 1:
        smooth = lambda data: [average_smoothing(series, smoothing_ws)
                               for series in data]

        test_accuracies = smooth(test_accuracies)
        if individual:
            test_losses = smooth(test_losses)
            train_losses = smooth(train_losses)

    # generate and save aggregated plot
    fig = plt.figure()

    for i in range(len(labels)):
        sns.lineplot(x=iterations[i], y=test_accuracies[i], label=labels[i])

    plt.xlabel('iteration')
    plt.ylabel('test accuracy')
    plt.title('Test accuracy over iterations')
    plt.legend()
    fig.tight_layout()

    os.makedirs(imagedir, exist_ok=True)
    plt.savefig(os.path.join(imagedir, '{}.png'.format(prefix)))

    # generate and save additional plots if needed
    if individual:
        for i in range(len(labels)):
            fig = plt.figure()

            sns.lineplot(x=iterations[i], y=train_losses[i], label='train')
            sns.lineplot(x=iterations[i], y=test_losses[i], label='test')

            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.title('Learning curve for {}'.format(labels[i]))
            plt.legend()
            fig.tight_layout()

            plt.savefig(os.path.join(imagedir, '{}_{}.png'.format(
                prefix, labels[i])))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group-prefix', '-gp', type=str, default='',
        help='Name prefix shared by all runs to be plotted.')
    parser.add_argument('--logdir', type=str, default='./logs',
        help='Directory containing logs of the runs.')
    parser.add_argument('--imagedir', type=str, default='./images',
        help='Directory where plot images are saved.')
    parser.add_argument('--individual', action='store_true',
        help='If set, learning curves are generated for all runs of the group.')
    parser.add_argument('--smoothing-window-size', '-ws', type=int, default=-1,
        help='Window size used to apply moving window smoothing to the data.')
    args = parser.parse_args()

    plot_group(args.group_prefix, args.logdir, args.imagedir,
        individual=args.individual, smoothing_ws=args.smoothing_window_size)

