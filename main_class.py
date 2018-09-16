from Entities import Board, StrategyEnum, Agent, Slot, Game
from random import randint
import random as rnd
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import SpanningTreeCoverage
import operator
import matplotlib.pyplot as plt
import itertools
import pylab
import os
import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
import time

So_seed = 123456789


def rotate(l, n):
    return l[n:] + l[:n]


def get_average_gain_no_information_job(board_size_x, board_size_y, seed_code):
    rnd.seed(seed_code)
    b = Board(board_size_x, board_size_y)

    agent_r = Agent("R", StrategyEnum.RandomSTC, randint(0, board_size_x - 1), randint(0, board_size_y - 1), board=b)
    print str(agent_r.InitPosX) + "," + str(agent_r.InitPosY)
    # add again randomness to the system
    rnd.seed(rnd.Random().random())

    agent_o = Agent("O", StrategyEnum.RandomSTC, randint(0, board_size_x - 1), randint(0, board_size_y - 1), board=b)
    print str(agent_o.InitPosX) + "," + str(agent_o.InitPosY)
    game = Game(b, agent_r, agent_o)
    game.run_game()
    gain = game.get_r_gain()

    if not (gain + game.get_o_gain() == board_size_x * board_size_y):
        print game.board
        print "Error: gains are wrong!"
    return gain


def get_average_gain_full_information_job(board_size_x, board_size_y):
    b = Board(board_size_x, board_size_y)

    o_init_size_x = randint(0, board_size_x - 1)
    o_init_size_y = randint(0, board_size_y - 1)
    agent_o_pos = (o_init_size_x, o_init_size_y)

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
    strategy = StrategyEnum.HorizontalCoverageCircular
    agent_o = Agent("O", strategy, agent_o_pos[0], agent_o_pos[1])
    agent_r = Agent("R", StrategyEnum.FullKnowledgeInterceptionCircular, agent_r_pos[0], agent_r_pos[1])
    game = Game(b, agent_r, agent_o)
    game.run_game()
    gain = game.get_r_gain()

    assert (gain + game.get_o_gain() == board_size_x * board_size_y)

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
    agentO = Agent("O", StrategyEnum.RandomSTC, randint(0, board_size - 1), randint(0, board_size - 1), board=b)
    game = Game(b, agent_r, agentO)
    game.run_game()
    return game.get_r_gain()


def get_mean_gain_set__sr__ir__so_random__io_job(agent_r, i_o, b):
    # Set randomness according to So_seed
    rnd.seed(So_seed)

    # Explanation:
    # we initiate the opponent at (0,0), so all opponents will have the save coverage path, but will start from
    # different positions.
    # Then, we rotate the steps list until opponent start from initial position. This way, all the opponents will have
    # the save coverage path, but each one will start from different location.
    agent_o = Agent("O", StrategyEnum.RandomSTC, 0, 0, board=b)
    initial_slot_index = agent_o.steps.index(Slot(i_o.row, i_o.col))
    agent_o.InitPosX = i_o.row
    agent_o.InitPosY = i_o.col
    agent_o.steps = rotate(agent_o.steps, initial_slot_index)

    # Set randomness back
    rnd.seed(os.urandom(100))
    game = Game(Board(b.Rows, b.Cols), agent_r, agent_o)
    game.run_game()
    gain = game.get_r_gain()
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
        agent_r = Agent("R", StrategyEnum.RandomSTC, int(board_size / 2), int(board_size / 2), board=b)

        rnd.seed(rnd.Random().random())

        set_r_gain = get_mean_gain_set__sr__ir_random__so__io(iterations, board_size, agent_r, b)
        mean_gains[seed_code] = set_r_gain
        print "seed_code: " + str(seed_code) + ", value: " + str(set_r_gain)

    print mean_gains

    rnd.seed(max(enumerate(mean_gains))[0])

    SpanningTreeCoverage.get_random_coverage_strategy(board_size, Slot(int(board_size / 2), int(board_size / 2)), True)
    return max(mean_gains)


def compute_expected_profits_given_o(board_size, seeds_count_o, given_io, given_ir, figure_label="", seeds_count_r=1,
                                     r_strategy=StrategyEnum.RandomSTC, r_base_seed=-1):
    """
    This function checks if there is a better-than-random strategy for R given only O's position and not its strategy.
    :param figure_label: optional figure's label
    :param given_ir: given initial position for r
    :param given_io: given initial position for o
    :param r_base_seed: optional user-given seed for Sr
    :type board_size: int
    :type seeds_count_o: int
    :type seeds_count_r: int
    """
    s_r_seeds = rnd.sample(xrange(1, 1000000000), seeds_count_r) if r_base_seed == -1 else [r_base_seed]
    s_o_seeds = rnd.sample(xrange(1, 1000000000), seeds_count_o)

    # save max r_code to print in the end
    max_val = (-1, -1)
    values = []
    # Run over a list of strategies for R
    for Sr_seed_code in s_r_seeds:
        b = Board(board_size, board_size)

        # Set seed for S_r
        rnd.seed(Sr_seed_code)
        agent_r = Agent("R", r_strategy, given_ir[0], given_ir[1], board=b)

        # Average over all possible strategies
        sum_gains = 0
        for So_seed_code in s_o_seeds:
            rnd.seed(So_seed_code)
            agent_o = Agent("O", StrategyEnum.RandomSTC, given_io[0], given_io[1], board=b)
            game = Game(Board(b.Rows, b.Cols), agent_r, agent_o)
            game.run_game()
            gain = game.get_r_gain()
            sum_gains += gain

        values.append(sum_gains / seeds_count_o)
        if sum_gains / seeds_count_o > max_val[0]:
            max_val = (sum_gains / seeds_count_o, Sr_seed_code)

    rnd.seed(max_val[1])
    SpanningTreeCoverage.get_random_coverage_strategy(board_size, Slot(given_ir[0], given_ir[1]),
                                                      Slot(given_io[0], given_io[1]),
                                                      print_mst=True, figure_label=figure_label)
    for i in xrange(len(s_r_seeds)):
        print "(seed, value, distance): {0} {1} {2}".format(str(s_r_seeds[i]), str(values[i]), str(
            np.fabs(given_ir[0] - given_io[0]) + np.fabs(given_ir[1] - given_io[1])))

    return s_r_seeds, values, max_val


def compute_expected_profits_for__sr_set__ir__so_random__io(board_size, seeds_count):
    seeds = range(0, seeds_count)

    b = Board(board_size, board_size)
    mean_gains = {}
    for seed_code in seeds:
        rnd.seed(seed_code)
        agent_r = Agent("R", StrategyEnum.RandomSTC, int(board_size / 2), int(board_size / 2), board=b)
        # rnd.seed(rnd.Random().random())
        rnd.seed(os.urandom(100))

        set_r_gain = get_mean_gain_set__sr__ir__so_random__io(board_size, agent_r, b)
        mean_gains[seed_code] = set_r_gain
        print "seed_code: {0}, value: {1}".format(str(seed_code), str(set_r_gain))

    max_seed = max(mean_gains.iteritems(), key=operator.itemgetter(1))[0]
    print "max_seed : " + str(max_seed)
    rnd.seed(max_seed)
    agent_r = Agent("R", StrategyEnum.RandomSTC, int(board_size / 2), int(board_size / 2), board=b)
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
    for i in xrange(len(path) - 2):
        p1 = path[i]
        p2 = path[i + 1]
        p3 = path[i + 2]
        if p1.go_east() == p2 and p2.go_east() != p3:
            turns += 1
            continue
        if p1.go_north() == p2 and p2.go_north() != p3:
            turns += 1
            continue
        if p1.go_south() == p2 and p2.go_south() != p3:
            turns += 1
            continue
        if p1.go_west() == p2 and p2.go_west() != p3:
            turns += 1
            continue

    return turns
    # SpanningTreeCoverage.display_path(path)


def get_heat_map_intersection_value(path_seed, i_o, i_r):
    rnd.seed(path_seed)
    path = SpanningTreeCoverage.get_random_coverage_strategy(32, Slot(i_r.row, i_r.col))
    hm_value = 0
    for time_t in xrange(len(path)):
        p_t = path[time_t]
        euclidean_dist = np.fabs(p_t.row - i_o.row) + np.fabs(p_t.col - i_o.col)
        if euclidean_dist > time_t:
            continue
        else:
            hm_value += np.power(1.1, time_t - euclidean_dist)
    return hm_value


def analyze_results(seeds, values, i_o, i_r, figure_label=""):
    f, ax = plt.subplots(1)

    # turns_values = [get_turns_amount(v) for v in seeds]
    # norm_turns_values = [float(i) / max(turns_values) for i in turns_values]
    # for t in norm_turns_values:
    #     print t
    # plt.plot(values, norm_turns_values, 'ro')

    hm_values = [get_heat_map_intersection_value(v, i_o, i_r) for v in seeds]
    # norm_hm_values = [float(i) / max(hm_values) for i in hm_values]
    ax.plot(values, hm_values, 'bo')

    # compute and show regression line
    if len(hm_values) > 3:
        m, b = pylab.polyfit(values, hm_values, 1)
        yp = pylab.polyval([m, b], values)
        ax.plot(values, yp, 'r')
    plt.grid('on')

    # display axis titles
    plt.xlabel('Average FCC')
    plt.ylabel('Heat-Map Value')

    if not os.path.exists(figure_label):
        os.makedirs(figure_label)

    f.savefig(figure_label + '/fcc_hmValues_graph.png', bbox_inches='tight')
    plt.close('all')


def send_files_via_email(text, title, file_name):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("moshe.samson@mail.huji.ac.il", "moshe_samson770")

    msg = MIMEMultipart()
    msg['From'] = "moshe.samson@mail.huji.ac.il"
    msg['To'] = COMMASPACE.join("samson.moshe@gmail.com")
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = title

    msg.attach(MIMEText(text))

    with open(file_name, "rb") as fil:
        part = MIMEApplication(
            fil.read(),
            Name=basename(file_name)
        )

    # After the file is closed
    part['Content-Disposition'] = 'attachment; filename="%s"' % basename(file_name)
    msg.attach(part)
    server.sendmail("moshe.samson@mail.huji.ac.il", "samson.moshe@gmail.com", msg.as_string())
    server.quit()


def compute_and_analyze_results_given_o_position(seeds_amount, i_o, _board_size, i_r, send_email):
    t0 = time.time()
    label = "Results/size{0}_Io={1}_Ir={2}_seeds:{3}".format(str(_board_size), str(i_o), str(i_r), str(seeds_amount))
    seeds, values, max_val = compute_expected_profits_given_o(board_size=_board_size, seeds_count_o=seeds_amount,
                                                              given_io=i_o, given_ir=i_r, figure_label=label,
                                                              r_strategy=StrategyEnum.RandomSTC,
                                                              seeds_count_r=1, r_base_seed=10)

    analyze_results(seeds, values, i_o=Slot(i_o[0], i_o[1]), i_r=Slot(i_r[0], i_r[1]), figure_label=label)
    t1 = time.time()
    # print "elapsed: " + str(t1 - t0)

    if send_email:
        send_files_via_email(text="Execution took " + str(t1 - t0) + " seconds. max_val: " + str(max_val), title=label)

        # print "\n==================================================================================================\n"


def is_there_best_strategy_r_only_positions(averaging_size = 50):
    """
    This method tries to find whether there is a better-than-other Sr.
    Averages over <averaging_size> different 'So's, and for each one check our X options for Sr, finding the average fcc
    Do that for Y pairs of Ir-Io, trying to prove optimality over different starting positions.

    The assumption, based on tests performed here, is that there is some better-than-others strategy So*. Now we try to
    show, for different strategies, that there is a better-than-others Sr, which we know the heuristic of it (not always
     stc).
    :return:
    """
    board_size = 50
    counter_rows = 0

    data_file = open("data.csv", 'a')
    data_file.write(",".join(["seed", "ir[0]", "ir[1]", "io[0]", "io[1]", "E[random]", "E[Spiraling in]",
                              "E[Spiraling out]", "E[Smart vertical]", "E[Smart semi circle]", "E[Smart cycle-out Io]"]))
    data_file.write("\n")

    while counter_rows < 1000:
        seed = rnd.random()
        rnd.seed(seed)

        # randomly select two different initial positions
        poss_positions = list(itertools.product(xrange(0, board_size - 1), xrange(0, board_size - 1)))
        ir, io = rnd.sample(poss_positions, 2)

        board = Board(board_size, board_size)

        # dumb R agents
        agent_r_random = Agent("R", StrategyEnum.RandomSTC, ir[0], ir[1], board=board)
        agent_r_spiraling_out = Agent("R", StrategyEnum.SpiralingOut, ir[0], ir[1], board=board)
        agent_r_spiraling_in = Agent("R", StrategyEnum.SpiralingIn, ir[0], ir[1], board=board)

        # sums
        random_sum = 0
        spiraling_out_sum = 0
        spiraling_in_sum = 0
        smart_vertical_sum = 0
        smart_semi_circle_sum = 0
        smart_cycle_out_io_sum = 0

        for i in xrange(averaging_size):
            print i
            rnd.seed(i)
            agent_o = Agent("O", StrategyEnum.RandomSTC, io[0], io[1], board=board)

            # smart R agents
            agent_r_vertical_far = Agent("R", StrategyEnum.VerticalFromFarthestCorner_OpponentAware, ir[0], ir[1],
                                         board=board, agent_o=agent_o)
            agent_r_semi_circle = Agent("R", StrategyEnum.SemiCyclingFromFarthestCorner_OpponentAware, ir[0], ir[1],
                                        board=board, agent_o=agent_o)
            agent_r_cycle_outside_io = Agent("R", StrategyEnum.CircleOutsideFromIo, ir[0], ir[1],
                                             board=board, agent_o=agent_o)

            # create and run games
            game = Game(Board(board.Rows, board.Cols), agent_r_random, agent_o)
            game.run_game()
            random_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r_spiraling_in, agent_o)
            game.run_game(optimality=False)
            spiraling_in_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r_spiraling_out, agent_o)
            game.run_game(optimality=False)
            spiraling_out_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r_vertical_far, agent_o)
            game.run_game(optimality=False)
            smart_vertical_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r=agent_r_semi_circle, agent_o=agent_o)
            game.run_game(optimality=False)
            smart_semi_circle_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r=agent_r_cycle_outside_io, agent_o=agent_o)
            game.run_game(optimality=False)
            smart_cycle_out_io_sum += game.get_r_gain()

        # print averaged data
        data_file = open("data.csv", 'a')
        data_file.write(",".join([str(seed), str(ir[0]), str(ir[1]), str(io[0]), str(io[1]),
                                  str(1.0 * random_sum / averaging_size),
                                  str(1.0 * spiraling_in_sum / averaging_size),
                                  str(1.0 * spiraling_out_sum / averaging_size),
                                  str(1.0 * smart_vertical_sum / averaging_size),
                                  str(1.0 * smart_semi_circle_sum / averaging_size),
                                  str(1.0 * smart_cycle_out_io_sum / averaging_size)]))
        data_file.write("\n")
        data_file.close()
        counter_rows += 1

        if counter_rows % 49 == 0:
            send_files_via_email("", "Simulations results", "data.csv")


def main():
    is_there_best_strategy_r_only_positions()        
                

if __name__ == "__main__":
    main()
