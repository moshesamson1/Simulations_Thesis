from threading import _BoundedSemaphore

from Entities import *
from random import randint
import random as rnd
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import SpanningTreeCoverage
import time
import os
import operator
import matplotlib
import matplotlib.pyplot as plt
from pylab import *

So_seed = 123456789


def rotate(l, n):
    return l[n:] + l[:n]


def get_average_gain_no_information_job(board_size_x, board_size_y, seed_code):
    rnd.seed(seed_code)
    b = Board(board_size_x, board_size_y)

    agent_r = Agent("R", "random", randint(0, board_size_x - 1), randint(0, board_size_y - 1), board=b)
    print str(agent_r.InitPosX) + "," + str(agent_r.InitPosY)
    # add again randomness to the system
    rnd.seed(rnd.Random().random())

    agent_o = Agent("O", "random", randint(0, board_size_x - 1), randint(0, board_size_y - 1), board=b)
    print str(agent_o.InitPosX) + "," + str(agent_o.InitPosY)
    game = Game(b, agent_r, agent_o)
    game.RunGame()
    gain = game.GetRGain()

    if not (gain + game.GetOGain() == board_size_x * board_size_y):
        print game._board
        print "Error: gains are wrong!"
    return gain


def get_average_gain_full_information_job(board_size_x, board_size_y):
    b = Board(board_size_x, board_size_y)

    o_init_size_x = randint(0, board_size_x - 1)
    o_init_size_y = randint(0, board_size_y - 1)
    agent_o_pos = (o_init_size_x, o_init_size_y);

    r_init_size_x = randint(0, board_size_x - 1)
    r_init_size_y = randint(0, board_size_y - 1)
    while (r_init_size_x, r_init_size_y) == (o_init_size_x, o_init_size_y):
        r_init_size_x = randint(0, board_size_x - 1)
        r_init_size_y = randint(0, board_size_y - 1)

    agent_r_pos = (r_init_size_x, r_init_size_y)
    # rnd.seed(5)

    # if rnd.uniform(0, 1) < 0.5:
    #    strategy = "optimalA"
    # else:
    strategy = "HorizontalCoverage"
    agent_o = Agent("O", strategy, agent_o_pos[0], agent_o_pos[1])
    agent_r = Agent("R", "FullKnowledgeInterception", agent_r_pos[0], agent_r_pos[1])
    game = Game(b, agent_r, agent_o)
    game.RunGame()
    gain = game.GetRGain()

    assert (gain + game.GetOGain() == board_size_x * board_size_y)

    return gain


def get_average_gain_full_information(iterations, board_size_x, board_size_y):
    num_cores = multiprocessing.cpu_count()
    gains = Parallel(n_jobs=num_cores - 1)(
        delayed(get_average_gain_full_information_job)(board_size_x, board_size_y) for i in xrange(0, iterations))
    return np.mean(gains)


def get_average_gain_no_information(iterations, board_size_x, board_size_y):
    num_cores = multiprocessing.cpu_count()
    gains = Parallel(n_jobs=num_cores - 1)(
        delayed(get_average_gain_no_information_job)(board_size_x, board_size_y, 1) for i in xrange(0, iterations))
    return np.mean(gains)


def get_mean_gain_set__sr__ir_random__so__io_job(board_size, agent_r, b):
    agentO = Agent("O", "random", randint(0, board_size - 1), randint(0, board_size - 1), board=b)
    game = Game(b, agent_r, agentO)
    game.RunGame()
    return game.GetRGain()


def get_mean_gain_set__sr__ir__so_random__io_job(agent_r, i_o, b):
    # Set randomness according to So_seed
    rnd.seed(So_seed)

    # Explanation:
    # we initiate the opponent at (0,0), so all opponents will have the save coverage path, but will start from
    # different positions.
    # Then, we rotate the steps list until opponent start from initial position. This way, all the opponents will have
    # the save coverage path, but each one will start from different location.
    agent_o = Agent("O", "random", 0, 0, board=b)
    initial_slot_index = agent_o.steps.index(Slot(i_o.row, i_o.col))
    agent_o.InitPosX = i_o.row
    agent_o.InitPosY = i_o.col
    agent_o.steps = rotate(agent_o.steps, initial_slot_index)

    # Set randomness back
    rnd.seed(os.urandom(100))
    game = Game(Board(b.Rows, b.Cols), agent_r, agent_o)
    game.RunGame()
    gain = game.GetRGain()
    # print "init: " + str(i_o) + ", gain: " + str(gain)
    return gain


def get_mean_gain_set__sr__ir_random__so__io(iterations, board_size, agent_r, b):
    num_cores = multiprocessing.cpu_count()
    gains = Parallel(n_jobs=num_cores - 1)(
        delayed(get_mean_gain_set__sr__ir_random__so__io_job)(board_size, agent_r, b) for i in xrange(0, iterations))
    return np.mean(gains)


def get_mean_gain_set__sr__ir__so_random__io(board_size, agent_r, b):
    # Using parallelism: works incorrectly with using random functions!
    # num_cores = multiprocessing.cpu_count()
    # gains = Parallel(n_jobs=num_cores - 1)(
    # delayed(get_mean_gain_set_Sr_Ir_So_random_Io_job)(agent_r, Slot(i,j), b) for i in xrange(0, board_size)
    # for j in xrange(0,board_size))

    gains = []
    for i in xrange(0, board_size):
        for j in xrange(0, board_size):
            #
            gains.append(get_mean_gain_set__sr__ir__so_random__io_job(agent_r, Slot(i, j), b))
    # print gains
    return np.mean(gains)


def compute_expected_profits_for__sr_set__ir_random__so__io(iterations, board_size, seeds_count):
    seeds = range(0, seeds_count)
    mean_gains = {}

    for seed_code in seeds:
        rnd.seed(seed_code)
        b = Board(board_size, board_size)
        agent_r = Agent("R", "random", int(board_size / 2), int(board_size / 2), board=b)

        rnd.seed(rnd.Random().random())

        set_r_gain = get_mean_gain_set__sr__ir_random__so__io(iterations, board_size, agent_r, b)
        mean_gains[seed_code] = set_r_gain
        print "seed_code: " + str(seed_code) + ", value: " + str(set_r_gain)

    print mean_gains

    rnd.seed(max(enumerate(mean_gains))[0])
    SpanningTreeCoverage.get_random_coverage_strategy(board_size, Slot(agent_r.InitPosX, agent_r.InitPosY), True)
    return max(mean_gains)


def compute_expected_profits_given_o(board_size, seeds_count_o, given_io, given_ir, figure_label="", seeds_count_r=1):
    """
    This function checks if there is a better-than-random strategy for R given only O's position and not its strategy.
    :param given_io:
    :type board_size: int
    :type seeds_count: int
    """
    s_r_seeds = rnd.sample(xrange(1, 1000000), seeds_count_r)
    s_o_seeds = rnd.sample(xrange(1, 1000000), seeds_count_o)

    # save max r_code to print in the end
    max_val = (-1, -1)
    values = []
    # Run over a list of strategies for R
    for Sr_seed_code in s_r_seeds:
        b = Board(board_size, board_size)

        # Set seed for S_r
        rnd.seed(Sr_seed_code)
        agent_r = Agent("R", "random", given_ir[0], given_ir[1], board=b)

        # Average over all possible strategies
        sum_gains = 0
        for So_seed_code in s_o_seeds:
            rnd.seed(So_seed_code)
            agent_o = Agent("O", "random", given_io[0], given_io[1], board=b)
            game = Game(Board(b.Rows, b.Cols), agent_r, agent_o)
            game.RunGame()
            gain = game.GetRGain()
            sum_gains += gain


        values.append(sum_gains/seeds_count_o)
        if sum_gains/seeds_count_o > max_val[0]:
            max_val = (sum_gains/seeds_count_o, Sr_seed_code)

    rnd.seed(max_val[1])
    SpanningTreeCoverage.get_random_coverage_strategy(board_size, Slot(given_ir[0], given_ir[1]), Slot(given_io[0], given_io[1]),
                                                      print_mst=True, figure_label=figure_label)
    for i in xrange(len(s_r_seeds)):
        print str(s_r_seeds[i]) + " " + str(values[i])

    return s_r_seeds, values


def compute_expected_profits_for__sr_set__ir__so_random__io(board_size, seeds_count):
    seeds = range(0, seeds_count)
    # means_by_So = {}

    # for i in So_seeds:
    #     So_seed = i

    mean_gains = {}
    for seed_code in seeds:
        b = Board(board_size, board_size)
        rnd.seed(seed_code)
        agent_r = Agent("R", "random", int(board_size / 2), int(board_size / 2), board=b)
        # rnd.seed(rnd.Random().random())
        rnd.seed(os.urandom(100))

        set_r_gain = get_mean_gain_set__sr__ir__so_random__io(board_size, agent_r, b)
        mean_gains[seed_code] = set_r_gain
        print "seed_code: " + str(seed_code) + ", value: " + str(set_r_gain)

    # means_by_So[So_seed] = mean_gains
    # print means_by_So

    max_seed = max(mean_gains.iteritems(), key=operator.itemgetter(1))[0]
    print "max_seed : " + str(max_seed)
    rnd.seed(max_seed)
    agent_r = Agent("R", "random", int(board_size / 2), int(board_size / 2), board=b)
    SpanningTreeCoverage.get_random_coverage_strategy(board_size, Slot(agent_r.InitPosX, agent_r.InitPosY), True)
    return max(mean_gains)


def get_turns_amount(path_seed):
    """
    Returns the amount of 'turns' in a single path
    :param path_seed: The path' seed
    :return: number of turns per path
    """
    rnd.seed(path_seed)
    path = SpanningTreeCoverage.get_random_coverage_strategy(32, Slot(31, 31))
    turns = 0
    for i in xrange(len(path)-2):
        p1 = path[i]
        p2 = path[i + 1]
        p3 = path[i + 2]
        if p1.GoRight() == p2 and p2.GoRight() != p3:
            turns += 1
            continue
        if p1.GoUp() == p2 and p2.GoUp() != p3:
            turns += 1
            continue
        if p1.GoDown() == p2 and p2.GoDown() != p3:
            turns += 1
            continue
        if p1.GoLeft() == p2 and p2.GoLeft() != p3:
            turns += 1
            continue

    return turns
    # SpanningTreeCoverage.display_path(path)


def get_heatmap_intersection_value(path_seed, i_o, i_r):
    rnd.seed(path_seed)
    path = SpanningTreeCoverage.get_random_coverage_strategy(32, Slot(i_r.row, i_r.col))
    hm_value = 0
    for time_t in xrange(len(path)):
        p_t = path[time_t]
        eucl_dist = np.fabs(p_t.row - i_o.row) + np.fabs(p_t.col - i_o.col)
        if eucl_dist > time_t:
            continue
        else:
            # if time_t-eucl_dist > 63: print "AHHHHHHH!!!!!"
            hm_value += np.power(1.1, time_t-eucl_dist)
    return hm_value


def analayze_results(seeds, values, figure_label=""):
    f, ax = plt.subplots(1)

    # turns_values = [get_turns_amount(v) for v in seeds]
    # norm_turns_values = [float(i) / max(turns_values) for i in turns_values]
    # for t in norm_turns_values:
    #     print t
    # plt.plot(values, norm_turns_values, 'ro')

    hm_values = [get_heatmap_intersection_value(v, Slot(16, 16), Slot(31, 31)) for v in seeds]
    norm_hm_values = [float(i) / max(hm_values) for i in hm_values]
    ax.plot(values, hm_values, 'bo')

    # compute and show regression line
    (m, b) = polyfit(values, hm_values, 1)
    yp = polyval([m, b], values)
    ax.plot(values, yp, 'r')
    plt.grid('on')

    # display axis titles
    plt.xlabel('Average FCC')
    plt.ylabel('Heat-Map Value')

    f.savefig(figure_label + '_fcc_hmValues_graph.png', bbox_inches='tight')
    plt.close('all')


def send_files_via_email(text, title):
    import smtplib
    from os.path import basename
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.utils import COMMASPACE, formatdate

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("moshe.samson@mail.huji.ac.il", "moshe_samson770")

    msg = MIMEMultipart()
    msg['From'] = "moshe.samson@mail.huji.ac.il"
    msg['To'] = COMMASPACE.join("samson.moshe@gmail.com")
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = title

    msg.attach(MIMEText(text))

    for f in ["max_path.png", "fcc_hmValues_graph.png"]:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)
    server.sendmail("moshe.samson@mail.huji.ac.il", "samson.moshe@gmail.com", msg.as_string())
    server.quit()


def compute_and_analyze_results_given_o_position(seeds_amount, i_o, _board_size, i_r, send_email):

    t0 = time.time()
    label = "size"+str(_board_size) + ", Io="+str(i_o) + ", Ir=" + str(i_r) + ", seeds: " + str(seeds_amount)
    seeds, values = compute_expected_profits_given_o(board_size=_board_size, seeds_count_o=seeds_amount, given_io=i_o,
                                                     given_ir=i_r, figure_label=label)
    print seeds
    print values
    analayze_results(seeds, values, figure_label=label)
    t1 = time.time()
    print "elapsed: " + str(t1 - t0)

    if send_email:
        send_files_via_email(text="Execution took " + str(t1 - t0) + " seconds.",
                         title="size32, Io=(16,16), Ir=(31,31), seeds: " + str(seeds_amount))

    print "\n=======================================================================================================\n"


def main():
    # compute_and_analyze_results_given_o_position(seeds_amount = 100, i_o = (16, 16), _board_size = 32, i_r = (31,31))
    compute_and_analyze_results_given_o_position(seeds_amount = 1000, i_o = (0, 0), _board_size = 32, i_r = (31,31),
                                                 send_email = False)


if __name__ == "__main__":
    main()
