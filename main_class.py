from Simulations_Thesis.Entities import Board, StrategyEnum, Agent, Slot, Game
from random import randint
import random as rnd
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from Simulations_Thesis import SpanningTreeCoverage
import operator
import os
import time
import seaborn as sb
import matplotlib.pyplot as plt
import itertools

import SpanningTreeCoverage
from Entities import Board, StrategyEnum, Agent, Slot, Game, send_files_via_email, DisplayingClass

So_seed = 123456789


def rotate(l, n):
    return l[n:] + l[:n]


def get_average_gain_no_information_job(board_size_x, board_size_y, seed_code):
    rnd.seed(seed_code)
    b = Board(board_size_x, board_size_y)

    agent_r = Agent("R", StrategyEnum.RandomSTC, randint(0, board_size_x - 1), randint(0, board_size_y - 1), board=b)
    print(str(agent_r.InitPosX) + "," + str(agent_r.InitPosY))
    # add again randomness to the system
    rnd.seed(rnd.Random().random())

    agent_o = Agent("O", StrategyEnum.RandomSTC, randint(0, board_size_x - 1), randint(0, board_size_y - 1), board=b)
    print(str(agent_o.InitPosX) + "," + str(agent_o.InitPosY))
    game = Game(b, agent_r, agent_o)
    game.run_game()
    gain = game.get_r_gain()

    if not (gain + game.get_o_gain() == board_size_x * board_size_y):
        print("Error: gains are wrong!")
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
        delayed(get_average_gain_full_information_job)(board_size_x, board_size_y) for i in range(0, iterations))
    return np.mean(gains)


def get_average_gain_no_information(iterations, board_size_x, board_size_y):
    num_cores = multiprocessing.cpu_count()
    gains = Parallel(n_jobs=num_cores - 1)(
        delayed(get_average_gain_no_information_job)(board_size_x, board_size_y, 1) for i in range(0, iterations))
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
        delayed(get_mean_gain_set__sr__ir_random__so__io_job)(board_size, agent_r, b) for i in range(0, iterations))
    return np.mean(gains)


def get_mean_gain_set__sr__ir__so_random__io(board_size, agent_r, b):
    # Using parallelism: works incorrectly with using random functions!
    # num_cores = multiprocessing.cpu_count()
    # gains = Parallel(n_jobs=num_cores - 1)(
    # delayed(get_mean_gain_set_Sr_Ir_So_random_Io_job)(agent_r, Slot(i,j), b) for i in range(0, board_size)
    # for j in range(0,board_size))

    gains = []
    for i in range(0, board_size):
        for j in range(0, board_size):
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
        print("seed_code: " + str(seed_code) + ", value: " + str(set_r_gain))

    print(mean_gains)

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
    s_r_seeds = rnd.sample(range(1, 1000000000), seeds_count_r) if r_base_seed == -1 else [r_base_seed]
    s_o_seeds = rnd.sample(range(1, 1000000000), seeds_count_o)

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
    for i in range(len(s_r_seeds)):
        print("(seed, value, distance): {0} {1} {2}".format(str(s_r_seeds[i]), str(values[i]), str(
            np.fabs(given_ir[0] - given_io[0]) + np.fabs(given_ir[1] - given_io[1]))))

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
        print("seed_code: {0}, value: {1}".format(str(seed_code), str(set_r_gain)))

    max_seed = max(mean_gains.iteritems(), key=operator.itemgetter(1))[0]
    print("max_seed : " + str(max_seed))
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
    for i in range(len(path) - 2):
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
    for time_t in range(len(path)):
        p_t = path[time_t]
        euclidean_dist = np.fabs(p_t.row - i_o.row) + np.fabs(p_t.col - i_o.col)
        if euclidean_dist > time_t:
            continue
        else:
            hm_value += np.power(1.1, time_t - euclidean_dist)
    return hm_value


def compute_and_analyze_results_given_o_position(seeds_amount, i_o, _board_size, i_r, send_email):
    t0 = time.time()
    label = "Results/size{0}_Io={1}_Ir={2}_seeds:{3}".format(str(_board_size), str(i_o), str(i_r), str(seeds_amount))
    seeds, values, max_val = compute_expected_profits_given_o(board_size=_board_size, seeds_count_o=seeds_amount,
                                                              given_io=i_o, given_ir=i_r, figure_label=label,
                                                              r_strategy=StrategyEnum.RandomSTC,
                                                              seeds_count_r=1, r_base_seed=10)

    t1 = time.time()
    # print "elapsed: " + str(t1 - t0)

    # if send_email:
    #     send_files_via_email(text="Execution took " + str(t1 - t0) + " seconds. max_val: " + str(max_val), title=label)

        # print "\n==================================================================================================\n"


def is_there_best_strategy_r_only_positions(averaging_size = 100):
    """
    This method tries to find whether there is a better-than-random Sr.
    Averages over <averaging_size> different 'So's, and for each one check our X options for Sr, finding the average fcc
    Do that for Y pairs of Ir-Io, trying to prove optimality over different starting positions.

    The assumption, based on tests performed here, is that there is some better-than-others strategy So*. Now we try to
    show, for different strategies, that there is a better-than-others Sr, which we know the heuristic of it (not always
     stc).
    :return:
    """
    board_size = 100
    counter_rows = 0

    data_file = open("data.csv", 'a')
    data_file.write(",".join(["seed", "ir[0]", "ir[1]", "io[0]", "io[1]", "E[random]", "E[Spiraling in]",
                              "E[Spiraling out]", "E[Smart vertical]", "E[Smart semi circle]", "E[Smart cycle-out Io]"]))
    data_file.write("\n")

    while counter_rows < 100:
        seed = rnd.random()
        rnd.seed(seed)

        # randomly select two different initial positions
        poss_positions = list(itertools.product(range(0, board_size - 1), range(0, board_size - 1)))
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

        for i in range(averaging_size):
            print("iteration: " + str(i) + " in round " + str(counter_rows))
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
            game.run_game(enforce_paths_length=False)
            spiraling_in_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r_spiraling_out, agent_o)
            game.run_game(enforce_paths_length=False)
            spiraling_out_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r_vertical_far, agent_o)
            game.run_game(enforce_paths_length=False)
            smart_vertical_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r=agent_r_semi_circle, agent_o=agent_o)
            game.run_game(enforce_paths_length=False)
            smart_semi_circle_sum += game.get_r_gain()

            game = Game(Board(board.Rows, board.Cols), agent_r=agent_r_cycle_outside_io, agent_o=agent_o)
            game.run_game(enforce_paths_length=False)
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

        # if counter_rows % 49 == 0:
        #     send_files_via_email("", "Simulations results", "data.csv")


def search_for_best_strategy():
    """
    This method tries to find wither there are some better-than_random STC strategy. The best strategy covered so-far,
    cover from the center outside, yield results little above average. We want to know how far can we reach. That is,
    what is the best strategy when can hope to achieve.
    :return: tuple, indicating best strategy seed, and correspondent result.
    """

    board_size = 50
    data_file = open("data.csv", 'a')

    for seed in range(1000):
        # seed indicates the seed. For that specific seed, average over X iterations, and return the expected value,
        # when for all iterations, io starts at the same position, and create strategy randomly

        # randomly select two different initial positions
        poss_positions = list(itertools.product(range(0, board_size - 1), range(0, board_size - 1)))
        ir, io = rnd.sample(poss_positions, 2)

        board = Board(board_size, board_size)

        rnd.seed(seed)
        agent_r_random = Agent("R", StrategyEnum.RandomSTC, ir[0], ir[1], board=board)
        random_sum = 0
        for j in range(100):
            agent_o = Agent("O", StrategyEnum.RandomSTC, io[0], io[1], board=board)
            game = Game(Board(board.Rows, board.Cols), agent_r_random, agent_o)
            game.run_game()
            random_sum += game.get_r_gain()
        result = random_sum / 100.0

        print("seed: " + str(seed) + " -> " + str(result))
        data_file.write(",".join([str(seed), str(result)]))
        data_file.write("\n")
    data_file.close()


def check_best_strategy(seed):
    board_size = 50
    data_file = open("data.csv", 'a')

    for iteration in range(500):
        # randomly select two different initial positions
        poss_positions = list(itertools.product(range(0, board_size - 1), range(0, board_size - 1)))
        ir, io = rnd.sample(poss_positions, 2)
        board = Board(board_size, board_size)

        rnd.seed(seed)
        agent_r_random = Agent("R", StrategyEnum.RandomSTC, ir[0], ir[1], board=board)
        random_sum = 0

        for _ in range(100):
            agent_o = Agent("O", StrategyEnum.RandomSTC, io[0], io[1], board=board)
            game = Game(Board(board.Rows, board.Cols), agent_r_random, agent_o)
            game.run_game()
            random_sum += game.get_r_gain()

        result = random_sum / 100.0
        print("(" + str(iteration) + ")" + " ir: " + str(ir) + ", io: " + str(io) + ", result: " + str(result))


def analyze_leader_follower(s_leader_1, s_leader_2, s_follower, s_opp, probs=None):
    """
    Simulating Leader-Follower game, with multiple strategies for the leader, and a single option for the follower.
    todo: Switch from only two strategies to x strategies
    :param s_leader_1: leader's first covering path
    :param s_leader_2: leader's second covering path
    :param s_follower: Follower's covering path
    :param s_opp:
    :param probs:
    :return:
    """
    if probs is None:
        probs = [0.5, 0.5]
    b = Board(100, 100)
    leader_agent_1 = Agent("Leader", s_leader_1, 0, 0, board=b)
    leader_agent_2 = Agent("Leader", s_leader_2, 0, 0, board=b)
    follower_agent = Agent("Follower", s_follower, 0, 0, board=b, agent_o=
    Agent("ActingAgainstAgent", s_opp, 0, 0, board=b) if s_opp is not None else None)

    g = Game(leader_agent_1, follower_agent)
    r_gain_1, o_gain_1 = g.run_game()
    print("Leader's Reward (%s): %d, Follower's Reward (%s responding to %s): %d" % (s_leader_1.name, r_gain_1,
                                                                                     s_follower.name, s_opp.name,
                                                                                     o_gain_1))
    g = Game(leader_agent_2, follower_agent)
    r_gain_2, o_gain_2 = g.run_game()
    print("Leader's Reward (%s): %d, Follower's Reward (%s responding to %s): %d" % (s_leader_2.name, r_gain_2,
                                                                                     s_follower.name, s_opp.name,
                                                                                     o_gain_2))
    print("\tWeighted Average: {}".format(probs[0] * o_gain_1 + probs[1] * o_gain_2))
    return leader_agent_1, leader_agent_2, follower_agent


def compare_between_coverage_methods(leader_s1: StrategyEnum, leader_s2: StrategyEnum, follower_s: StrategyEnum) -> None:
    """
    #type:(int,int)
    :param leader_s1: First Strategy
    :param leader_s2: Second Strategy
    :param follower_s: Second Strategy
    :return: Nothing, for now
    """

    leader_agent_s1,leader_agent_s2, follower_agent = analyze_leader_follower(leader_s1, leader_s2, follower_s, leader_s1)

    # adjacent corner check
    _, _, follower_agent_rows = analyze_leader_follower(leader_s1, leader_s2, StrategyEnum.SemiCyclingFromAdjacentCorner_row_OpponentAware, leader_s1)
    _, _, follower_agent_cols = analyze_leader_follower(leader_s1, leader_s2, StrategyEnum.SemiCyclingFromAdjacentCorner_col_OpponentAware, leader_s1)


    # display heat maps
    a = Agent("cbq", StrategyEnum.QuartersCoverageCircular,0,0, Board(100,100))

    # display the two strategies the leader is considering, and the cross heatmap
    leader_agent_s1.display_heat_map(0, 0)
    leader_agent_s2.display_heat_map(0, 1)
    leader_s1_s2_cross_hm = leader_agent_s1.display_cross_heatmap(leader_agent_s2, display_grid_x=0, display_grid_y=2,
                                                                  probabilities=[0.5, 0.5])

    # display the follower possible response strategy
    follower_agent.display_heat_map(1, 0)
    follower_agent.display_sub_heatmap(leader_s1_s2_cross_hm, display_grid_x=1, display_grid_y=2,
                                       probabilities=[0.5, 0.5])

    follower_agent_rows.display_heat_map(2, 0)
    follower_agent_rows.display_sub_heatmap(leader_s1_s2_cross_hm, display_grid_x=2, display_grid_y=2,
                                            probabilities=[0.5, 0.5])

    follower_agent_cols.display_heat_map(3, 0)
    follower_agent_cols.display_sub_heatmap(leader_s1_s2_cross_hm, display_grid_x=3, display_grid_y=2,
                                            probabilities=[0.5, 0.5])

    # quarters
    a.display_heat_map(4,0)
    leader_agent_s2.display_heat_map(4, 1)
    a.display_cross_heatmap(leader_agent_s2, 4, 2, [0.5, 0.5])

    DisplayingClass.get_plt().show()

from tqdm import tqdm
def take_snapshots():
    num_samples=100

    random_results = []
    steps_times_o={}
    steps_times_r={}
    for i in range(1024):
        steps_times_o[i]=[]
    for i in range(1056):
        steps_times_r[i]=[]

    for _ in tqdm(range(num_samples)):
        b = Board(32, 32)
        agentO = Agent("O", StrategyEnum.RandomSTC, 31, 31, board=b)
        agentR = Agent("R", StrategyEnum.LONGEST_TO_REACH, 0, 0, board=b, agent_o=agentO)
        game = Game(b, agentR, agentO)
        game.run_game(enforce_paths_length=False)
        for os in range(len(agentO.steps)):
            step = agentO.steps[os]
            steps_times_o[os].append(step)

        random_results.append(game.get_r_gain())


    b = Board(32, 32)
    agentO = Agent("O", StrategyEnum.RandomSTC, 31, 31, board=b)
    agentR = Agent("R", StrategyEnum.LONGEST_TO_REACH, 0, 0, board=b, agent_o=agentO)
    for os in range(len(agentR.steps)):
        step = agentR.steps[os]
        steps_times_r[os].append(step)

    from collections import Counter
    iterations = float(num_samples)
    to_t = []

    # for cell c, how much strategies cover it by time t\
    for t in tqdm(range(1024)):
        to_t.extend(steps_times_o[t])
        c=Counter(to_t)
        probes = [[c[Slot(i, j)] / iterations for j in range(32)] for i in range(32)]
        probes.reverse()
        sb.heatmap(probes, yticklabels=False, xticklabels=False)
        plt.savefig('data_O/time_%s'%t)
        plt.close()

    # for t in tqdm(range(1056)):
    #     to_t.extend(steps_times_r[t])
    #     c=Counter(to_t)
    #     probes = [[c[Slot(i, j)] / iterations for j in range(32)] for i in range(32)]
    #     probes.reverse()
    #     sb.heatmap(probes,yticklabels=False, xticklabels=False)
    #     plt.savefig('data_R/time_%s'%t)
    #     plt.close()


    # print(steps_times_o)
    # print((sum(random_results) / float(len(random_results))) / 1024.0)

def validate_lcp():
    num_samples=100

    random_results = []
    for _ in tqdm(range(num_samples)):
        b = Board(32, 32)
        agentO = Agent("O", StrategyEnum.RandomSTC, 31, 31, board=b)
        agentR = Agent("R", StrategyEnum.RandomSTC, 0, 0, board=b, agent_o=agentO)
        game = Game(b, agentR, agentO)
        game.run_game(enforce_paths_length=False)
        random_results.append(game.get_r_gain())
    print((sum(random_results) / float(len(random_results))) / 1024.0)

    lcp_results = []
    for _ in tqdm(range(num_samples)):
        b = Board(32,32)
        agentO = Agent("O",StrategyEnum.RandomSTC, 31, 31, board=b)
        agentR = Agent("R",StrategyEnum.LCP , 0, 0, board=b, agent_o=agentO)
        game = Game(b,agentR,agentO)
        game.run_game(enforce_paths_length=False)
        lcp_results.append(game.get_r_gain())
    print((sum(lcp_results)/float(len(lcp_results)))/1024.0)

    ltr_results = []
    for _ in tqdm(range(num_samples)):
        b = Board(32, 32)
        agentO = Agent("O", StrategyEnum.RandomSTC, 31, 31, board=b)
        agentR = Agent("R", StrategyEnum.LONGEST_TO_REACH, 0, 0, board=b, agent_o=agentO)
        game = Game(b, agentR, agentO)
        game.run_game(enforce_paths_length=False)
        ltr_results.append(game.get_r_gain())
    print((sum(ltr_results) / float(len(ltr_results))) / 1024.0)

    circVert_results = []
    for _ in tqdm(range(num_samples)):
        b = Board(32, 32)
        agentO = Agent("O", StrategyEnum.RandomSTC, 31, 31, board=b)
        agentR = Agent("R", StrategyEnum.VerticalCoverageCircular, 0, 0, board=b, agent_o=agentO)
        game = Game(b, agentR, agentO)
        game.run_game(enforce_paths_length=False)
        circVert_results.append(game.get_r_gain())
    print((sum(circVert_results) / float(len(circVert_results))) / 1024.0)

    noncircVert_results = []
    for _ in tqdm(range(num_samples)):
        b = Board(32, 32)
        agentO = Agent("O", StrategyEnum.RandomSTC, 31, 31, board=b)
        agentR = Agent("R", StrategyEnum.VerticalCoverageNonCircular, 0, 0, board=b, agent_o=agentO)
        game = Game(b, agentR, agentO)
        game.run_game(enforce_paths_length=False)
        noncircVert_results.append(game.get_r_gain())
    print((sum(noncircVert_results) / float(len(noncircVert_results))) / 1024.0)

    circHor_results = []
    for _ in tqdm(range(num_samples)):
        b = Board(32, 32)
        agentO = Agent("O", StrategyEnum.RandomSTC, 31, 31, board=b)
        agentR = Agent("R", StrategyEnum.HorizontalCoverageCircular, 0, 0, board=b, agent_o=agentO)
        game = Game(b, agentR, agentO)
        game.run_game(enforce_paths_length=False)
        circHor_results.append(game.get_r_gain())
    print((sum(circHor_results) / float(len(circHor_results))) / 1024.0)

    print("full data: ")
    for l in zip(random_results, lcp_results, ltr_results, circVert_results, noncircVert_results, circHor_results):
        print(str(l)[1:-1])

def check_tdv():
    b = Board(32, 32)
    agentO = Agent("O", StrategyEnum.RandomSTC, 31,31, board=b)
    agentR = Agent("R", StrategyEnum.RandomSTC, 0, 0, board=b, agent_o=agentO)
    print(agentR.get_tdv())

    b = Board(32, 32)
    agentO = Agent("O", StrategyEnum.RandomSTC, 31,31, board=b)
    agentR = Agent("R", StrategyEnum.LCP, 0, 0, board=b, agent_o=agentO)
    print(agentR.get_tdv())

    b = Board(32, 32)
    agentO = Agent("O", StrategyEnum.RandomSTC, 31,31, board=b)
    agentR = Agent("R", StrategyEnum.LONGEST_TO_REACH, 0, 0, board=b, agent_o=agentO)
    print(agentR.get_tdv())

def main():
    # compare_between_coverage_methods(StrategyEnum.VerticalCoverageCircular,
    #                                  StrategyEnum.HorizontalCoverageCircular,
    #                                  StrategyEnum.SemiCyclingFromFarthestCorner_OpponentAware)
    # take_snapshots()
    validate_lcp()
    # check_tdv()

if __name__ == "__main__":
    main()
