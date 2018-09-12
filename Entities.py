import numpy as np
import SpanningTreeCoverage
from math import *


class Board:

    def __init__(self, rows, cols):
        self.Rows = rows
        self.Cols = cols
        self.Slots = [[Slot(y, x) for x in range(self.Rows)] for y in range(self.Cols)]

    def __str__(self):
        s = ''
        for i in range(len(self.Slots)):
            for j in range(len(self.Slots[i])):
                s += str(self.Slots[i][j])
            s += '\n'
        return s

    def reset(self):
        for s in self.Slots:
            s.has_been_visited = False
            s.is_occupied = False
            s.covered_by = "*"


class Slot:
    def __init__(self, x, y):
        self.has_been_visited = False
        self.covered_by = "*"
        self.row = x
        self.col = y
        self.Name = "("+str(x) + "," + str(y) + ")"

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __ne__(self, other):
        return self.row != other.row or self.col != other.col

    def __hash__(self):
        return hash((self.row, self.col))

    def go_west(self):
        return Slot(self.row, self.col - 1)

    def go_east(self):
        return Slot(self.row, self.col + 1)

    def go_north(self):
        return Slot(self.row - 1, self.col)

    def go_south(self):
        return Slot(self.row + 1, self.col)

    def go(self, s, opposite_direction = False):
        """
        Go South, East, North or West, according to the given parameter.
        :param s: Must by 'd', 'r','n','l'
        :param opposite_direction: False by default. Move to the opposite direction
        :return: the new slot
        """
        assert s == 'u' or s == 'd' or s == 'r' or s == 'l'

        if opposite_direction:
            if s == 'u':
                s = 'd'
            if s == 'd':
                s = 'u'
            if s == 'r':
                s = 'l'
            if s == 'l':
                s = 'r'

        if s == 'u':
            return self.go_north()
        if s == 'd':
            return self.go_south()
        if s == 'r':
            return self.go_east()
        if s == 'l':
            return self.go_west()

    def to_tuple(self):
        return self.row, self.col

    def __str__(self):
        return "{0},{1},{2}".format(str(int(self.row)), str(int(self.col)), str(self.covered_by))

    def __repr__(self):
        return str(self)


class StrategyEnum:
    def __init__(self):
        pass

    VerticalCoverageCircular, HorizontalCoverageCircular, FullKnowledgeInterceptionCircular, QuartersCoverageCircular,\
        RandomSTC, VerticalCoverageNonCircular, SpiralingOut, SpiralingIn, VerticalFromFarthestCorner_OpponentAware,\
        SemiCyclingFromFarthestCorner_OpponentAware, SpiralingOut_OpponentAware = range(11)





class Agent:
    def __init__(self, name, strategy, x, y, board=None, agent_o=None):
        # type: (str, int, int, int, Board, Agent) -> None
        assert isinstance(strategy, int)

        self.Name = name
        self.Strategy = strategy
        self.InitPosX = x
        self.InitPosY = y
        self.gameBoard = board

        if self.Strategy == StrategyEnum.VerticalCoverageCircular:
            self.steps = get_vertical_coverage_steps(self, len(self.gameBoard.Slots), len(self.gameBoard.Slots[0]))
        elif self.Strategy == StrategyEnum.HorizontalCoverageCircular:
            self.steps = get_horizontal_coverage_steps(self, len(self.gameBoard.Slots), len(self.gameBoard.Slots[0]))
        elif self.Strategy == StrategyEnum.FullKnowledgeInterceptionCircular:
            self.steps = run_agent_over_board_full_knowledge_interception_strategy(self, agent_o,
                                                                                   len(self.gameBoard.Slots),
                                                                                   len(self.gameBoard.Slots[0]))
        elif self.Strategy == StrategyEnum.QuartersCoverageCircular:
            self.steps = get_quarters_coverage_steps(self, len(self.gameBoard.Slots), len(self.gameBoard.Slots[0]))
        elif self.Strategy == StrategyEnum.RandomSTC:
            self.steps = SpanningTreeCoverage.get_random_coverage_strategy(len(self.gameBoard.Slots),
                                                                           Slot(self.InitPosX, self.InitPosY),
                                                                           print_mst=False)
        elif self.Strategy == StrategyEnum.VerticalCoverageNonCircular:
            self.steps = get_non_circular_vertical_coverage(self, len(self.gameBoard.Slots),
                                                            len(self.gameBoard.Slots[0]))
        elif self.Strategy == StrategyEnum.SpiralingIn:
            self.steps = get_spiraling_in_steps(self, len(self.gameBoard.Slots))
        elif self.Strategy == StrategyEnum.SpiralingOut:
            self.steps = get_spiraling_out_steps(self, len(self.gameBoard.Slots))
        elif self.Strategy == StrategyEnum.VerticalFromFarthestCorner_OpponentAware:
            self.steps = get_vertical_coverage_farthest_corner_opponent_aware_steps(self, len(self.gameBoard.Slots),
                                                                                    agent_o=agent_o)
        elif self.Strategy == StrategyEnum.SemiCyclingFromFarthestCorner_OpponentAware:
            self.steps = get_semi_cycle_coverage_farthest_corner_opponent_aware_steps(self, len(self.gameBoard.Slots),
                                                                                  agent_o=agent_o)
        elif self.Strategy == StrategyEnum.SpiralingOut_OpponentAware:
            self.steps = \
                get_spiraling_out_from_opp_opponent_aware(self, len(self.gameBoard.Slots), agent_o=agent_o)

class Game:

    def __init__(self, board, agent_r, agent_o):
        self._board = board
        self._agentR = agent_r
        self._agentO = agent_o

    def run_game(self, optimality=True):
        steps_r = self._agentR.steps
        steps_o = self._agentO.steps

        # print steps_o
        if optimality:
            assert len(steps_o) == len(steps_r)

        for i in range(min(len(steps_r), len(steps_o))):
            # perform step for R
            step_r = steps_r[i]
            if not self._board.Slots[int(step_r.row)][int(step_r.col)].has_been_visited:
                self._board.Slots[int(step_r.row)][int(step_r.col)].has_been_visited = True
                self._board.Slots[int(step_r.row)][int(step_r.col)].covered_by = self._agentR.Name

            # then perform step for O
            step_o = steps_o[i]
            if not self._board.Slots[int(step_o.row)][int(step_o.col)].has_been_visited:
                self._board.Slots[int(step_o.row)][int(step_o.col)].has_been_visited = True
                self._board.Slots[int(step_o.row)][int(step_o.col)].covered_by = self._agentO.Name



    def get_r_gain(self):
        cond_count = 0

        # print self._board.Slots

        for i in xrange(0, self._board.Rows):
            for j in xrange(0, self._board.Cols):
                if self._board.Slots[i][j].covered_by == self._agentR.Name:
                    cond_count += 1

        return float(cond_count)

    def get_o_gain(self):
        cond_count = 0

        size_x = len(self._board.Slots)
        size_y = len(self._board.Slots[0])

        for i in xrange(0, size_x):
            for j in xrange(0, size_y):
                if self._board.Slots[i][j].covered_by == self._agentO.Name:
                    cond_count += 1

        return float(cond_count)

    @property
    def board(self):
        return self._board


def run_agent_over_board_interception_strategy(steps_o, i_r_x, i_r_y, interception_point):
    distance2_interception_point = fabs(i_r_x - interception_point[0]) + fabs(i_r_y - interception_point[1])
    next_slot = (i_r_x, i_r_y)
    steps = [next_slot]
    counter = 0

    if np.sign(interception_point[0] - i_r_x) >= 0:
        x_sign = 1
    else:
        x_sign = -1

    if np.sign(interception_point[1] - i_r_y) >= 0:
        y_sign = 1
    else:
        y_sign = -1

    while True:
        if counter >= len(steps_o)-1:
            break

        # first, go to interception point
        if counter < distance2_interception_point:
            for x_step in xrange(i_r_x + x_sign, interception_point[0] + x_sign, x_sign):
                counter += 1
                next_slot = (x_step, next_slot[1])
                steps.append(next_slot)
            for y_step in xrange(i_r_y + y_sign, interception_point[1] + y_sign, y_sign):
                counter += 1
                next_slot = (next_slot[0], y_step)
                steps.append(next_slot)
        # then, play as O plays
        else:
            counter += 1
            next_slot = steps_o[counter]
            steps.append(next_slot)

    return steps


def get_interception_point(steps_o, i_r_x, i_r_y):
    ip_x = -1
    ip_y = -1

    steps_counter = 0
    for step in steps_o:
        steps_counter += 1
        distance_from_i_r = fabs(step[0] - i_r_x) + fabs(step[1] - i_r_y) + 1
        if fabs(steps_counter - distance_from_i_r) <= 1:
            ip_x, ip_y = step
            break

    assert (ip_x != -1 and ip_y != -1)

    return (ip_x, ip_y), fabs(ip_x - i_r_x) + fabs(ip_y - i_r_y)


def get_vertical_coverage_steps(agent, board_size_x, board_size_y):
    next_slot = (agent.InitPosX, agent.InitPosY)

    flag = (agent.InitPosY == board_size_y - 1 and not (agent.InitPosX == 0))

    steps = []
    counter = 0

    while True:
        counter += 1
        if counter > 1000000000000:
            break

        steps.append(next_slot)
        # in the middle of moving from bottom row to top row
        if flag:
            next_slot = (next_slot[0] - 1, next_slot[1])
            if next_slot[0] == 0:
                flag = False
            if next_slot == (agent.InitPosX, agent.InitPosY):
                break
            continue
        # check if in last position, and start moving from last row to top row
        elif next_slot[0] == board_size_x - 1 and next_slot[1] == board_size_y - 1 - 1:
            flag = True
            next_slot = (next_slot[0], next_slot[1] + 1)
            if next_slot == (agent.InitPosX, agent.InitPosY):
                break
            continue
        # update next slot
        elif next_slot[0] % 2 != 0:
            if next_slot[1] == board_size_y - 1 - 1:
                next_slot = (next_slot[0] + 1, next_slot[1])
            else:
                next_slot = (next_slot[0], next_slot[1] + 1)
        else:
            if next_slot[1] == 0:
                next_slot = (next_slot[0] + 1, next_slot[1])
            else:
                next_slot = (next_slot[0], next_slot[1] - 1)

        if next_slot == (agent.InitPosX, agent.InitPosY):
            break

    return steps


def get_non_circular_vertical_coverage(agent, board_size_x, board_size_y):
    """
    Returns a non-circular vertical coverage, starting from agent's initial position to top-right position, then cover
    all cells from top-right to bottom-left, without repeating cells (there are some assumption regarding the initial
    position
    :param agent: the given agent
    :param board_size_x: board's width
    :param board_size_y: board's height
    :return: a set of steps covering the whole board
    """
    steps = []
    next_slot = Slot(agent.InitPosX, agent.InitPosY)
    turning_slot = Slot(agent.InitPosX, agent.InitPosY)
    reaching_to_farthest_corner = True
    up_or_down = 'd'

    # assert call
    assert agent.InitPosX % 2 == 0
    while True:
        steps.append(next_slot)
        if len(steps) >= board_size_y*board_size_x:
            break

        # Check if we agent reached the farthest corner of the board
        if next_slot == Slot(board_size_x - 1, board_size_y - 1):
            reaching_to_farthest_corner = False

        if reaching_to_farthest_corner:
            if next_slot.row < board_size_y - 1:
                next_slot = next_slot.go_south()
                continue
            elif next_slot.col < board_size_x - 1:
                next_slot = next_slot.go_east()
                continue
        else:
            if next_slot.col > agent.InitPosY:
                if next_slot.col % 2 != 0:
                    next_slot = next_slot.go_north() if next_slot.row > 0 else next_slot.go_west()
                else:
                    next_slot = next_slot.go_south() if next_slot.row < board_size_y-2 else next_slot.go_west()
                continue
            else:
                if up_or_down == 'd':
                    if next_slot.go_south() != turning_slot:
                        if next_slot.row < board_size_y-1:
                            next_slot = next_slot.go_south()
                        else:
                            next_slot = next_slot.go_west()
                            up_or_down = 'u'
                        continue
                    else:
                        turning_slot = turning_slot.go_west().go_west().go_north().go_north()
                        steps.append(next_slot.go_west())
                        steps.append(next_slot.go_west().go_south())
                        next_slot = next_slot.go_west().go_south().go_south()
                        continue
                else:
                    if next_slot != turning_slot:
                        if next_slot.row > 0:
                            next_slot = next_slot.go_north()
                        else:
                            next_slot = next_slot.go_west()
                            up_or_down = 'd'
                        continue
                    else:
                        steps.append(next_slot.go_east())
                        steps.append(next_slot.go_east().go_north())
                        next_slot = next_slot.go_east().go_north().go_north()
                        continue

    return steps[:board_size_y*board_size_x]


def get_quarters_coverage_steps(agent, board_size_x, board_size_y):
    # This coverage strategy covers first the top left, then top right, then bottom left, then bottom right quarters of
    # the area.
    # While building this function, we assumed 100X100 dimensions

    next_slot = Slot(agent.InitPosX, agent.InitPosY)
    # flag = (agent.InitPosY == boardSizeY - 1 and not (agent.InitPosX == 0))

    steps = []
    counter = 0

    while True:
        counter += 1

        if counter > 100000:
            print "Something is wrong, counter is too big!"
            print steps
            break

        if counter > 1 and (next_slot.row, next_slot.col) == (agent.InitPosX, agent.InitPosY):
            break

        steps.append((next_slot.row, next_slot.col))

        # TL Quarter
        if 0 <= next_slot.row < board_size_x/2 and 0 <= next_slot.col < board_size_y/2:
            if (next_slot.row, next_slot.col) == (board_size_x / 2 - 1, board_size_y / 2 - 1):
                next_slot.row, next_slot.col = next_slot.go_east()
                continue
            if next_slot.col == 0:
                next_slot.row, next_slot.col = next_slot.go_north() if next_slot.row > 0 else next_slot.go_east()
                continue
            if next_slot.row % 2 == 0 and next_slot.row < board_size_x/2-2:  # An even line, not the last one
                next_slot.row, next_slot.col = next_slot.go_east() if not next_slot.col == board_size_y / 2 - 1 else \
                    next_slot.go_south()
                continue
            elif next_slot.row % 2 != 0 and not next_slot.row == board_size_x / 2 - 1:  # An odd line, not before last
                next_slot.row, next_slot.col = next_slot.go_west() if not next_slot.col == 1 else next_slot.go_south()
                continue
            elif next_slot.row % 2 == 0 and next_slot.row == board_size_x/2-2:  # An even line, the last one
                next_slot.row, next_slot.col = next_slot.go_south() if next_slot.col % 2 != 0 else next_slot.go_east()
            elif next_slot.row % 2 != 0 and next_slot.row == board_size_x/2-1:  # An odd line, last line
                next_slot.row, next_slot.col = next_slot.go_east() if next_slot.col % 2 != 0 else next_slot.go_north()
            else:
                print "TL: Error occurred! Should not reach here!"
            continue

        # TR Quarter
        elif 0 <= next_slot.row < board_size_x/2 and board_size_y/2 <= next_slot.col < board_size_y:
            if (next_slot.row, next_slot.col) == (board_size_x / 2 - 1, board_size_y - 1):
                next_slot.row, next_slot.col = next_slot.go_south()
                continue
            elif next_slot.col % 2 == 0:
                next_slot.row, next_slot.col = next_slot.go_north() if next_slot.row > 0 else next_slot.go_east()
                continue
            elif next_slot.col % 2 != 0:
                next_slot.row, next_slot.col = next_slot.go_south() if next_slot.row < board_size_x / 2 - 1 else \
                    next_slot.go_east()
                continue
            else:
                print "TR: Error occurred! Should not reach here!"
            continue

        # BL Quarter
        elif board_size_x/2 <= next_slot.row < board_size_x and 0 <= next_slot.col < board_size_y/2:
            if (next_slot.row, next_slot.col) == (board_size_x / 2, 0):  # last cell of quarter
                next_slot.row, next_slot.col = next_slot.go_north()
                continue
            elif next_slot.col % 2 == 0:  # an even column
                next_slot.row, next_slot.col = next_slot.go_north() if next_slot.row > board_size_x / 2 else \
                    next_slot.go_west()
                continue
            elif next_slot.col % 2 != 0:  # An odd line
                next_slot.row, next_slot.col = next_slot.go_south() if next_slot.row < board_size_x - 1 else \
                    next_slot.go_west()
                continue
            else:
                print "BL: Error occurred! Should not reach here!"
            continue

        # BR Quarter
        else:
            if (next_slot.row, next_slot.col) == (board_size_x / 2, board_size_y / 2):  # last cell of quarter
                next_slot.row, next_slot.col = next_slot.go_west()
                continue
            elif next_slot.col == board_size_y-1:
                next_slot.row, next_slot.col = next_slot.go_south() if not next_slot.row == board_size_x - 1 else \
                    next_slot.go_west()
                continue
            elif next_slot.row % 2 != 0 and not next_slot.row == board_size_x / 2 + 1:  # and odd line, not before last
                next_slot.row, next_slot.col = next_slot.go_west() if next_slot.col > board_size_y / 2 else \
                    next_slot.go_north()
                continue
            elif next_slot.row % 2 != 0 and next_slot.row == board_size_x/2+1:  # and odd line, DO before last
                next_slot.row, next_slot.col = next_slot.go_north() if next_slot.col % 2 == 0 else next_slot.go_west()
                continue
            elif next_slot.row % 2 == 0 and not next_slot.row == board_size_x / 2:  # an even line, not last one
                next_slot.row, next_slot.col = next_slot.go_east() if next_slot.col < board_size_y - 2 else \
                    next_slot.go_north()
                continue
            elif next_slot.row % 2 == 0 and next_slot.row == board_size_x/2:  # an even line, INDEED last one
                next_slot.row, next_slot.col = next_slot.go_west() if next_slot.col % 2 == 0 else next_slot.go_south()
                continue
            else:
                print "BR: Error occurred! Should not reach here!"
            continue

    return steps


def get_horizontal_coverage_steps(agent, board_size_x, board_size_y):
    next_slot = (agent.InitPosX, agent.InitPosY)
    flag = (agent.InitPosX == board_size_x - 1)

    steps = []
    counter = 0

    # print "init pos {},{}: ".format(agent.InitPosX, agent.InitPosY)

    while True:
        counter += 1
        if counter > board_size_x * board_size_y:
            break

        steps.append(next_slot)
        # in the middle of moving from bottom rpw to top row
        if flag:
            if next_slot[1] == board_size_y - 1:
                flag = False
                next_slot = (next_slot[0] - 1, next_slot[1])
            else:
                next_slot = (next_slot[0], next_slot[1] + 1)
            if next_slot == (agent.InitPosX, agent.InitPosY):
                break
            continue
        # check if in last position, and start moving from last row to top row
        elif next_slot[0] == board_size_x - 1 - 1 and next_slot[1] == 0:
            flag = True
            next_slot = (next_slot[0] + 1, next_slot[1])

            if next_slot == (agent.InitPosX, agent.InitPosY):
                break
            continue
        # update next slot
        elif next_slot[1] % 2 != 0:
            if next_slot[0] == 0:
                next_slot = (next_slot[0], next_slot[1] - 1)
            else:
                next_slot = (next_slot[0] - 1, next_slot[1])
        else:
            if next_slot[0] == board_size_y - 1 - 1:
                next_slot = (next_slot[0], next_slot[1] - 1)
            else:
                next_slot = (next_slot[0] + 1, next_slot[1])
                if next_slot[0] > board_size_x - 1:
                    print "+1"

        if next_slot == Slot(agent.InitPosX, agent.InitPosY):
            break

    return steps


def get_spiraling_in_steps(agent, board_size):
    """
    This function return the agent steps, when deciding to cover the world, spiraling from outside to inside.
    Note: this coverage method is not optimal!
    :param agent: the agent covering the world
    :param board_size: self explanatory
    :return: list of steps
    """
    steps = []

    next_slot = Slot(agent.InitPosX, agent.InitPosY)

    # start by going toward the closest corner
    if next_slot.row < board_size / 2:
        while next_slot.row > 0:
            steps.append(next_slot)
            next_slot = next_slot.go_north()
            continue
    else:
        while next_slot.row < board_size - 1:
            steps.append(next_slot)
            next_slot = next_slot.go_south()
            continue

    if next_slot.col < board_size / 2:
        while next_slot.col > 0:
            steps.append(next_slot)
            next_slot = next_slot.go_west()
            continue
    else:
        while next_slot.col < board_size - 1:
            steps.append(next_slot)
            next_slot = next_slot.go_east()
            continue
    steps.append(next_slot)

    # after reaching the closest-to-start corner, start covering the world, counter clockwise
    shallow_slots = [[0 for x in range(board_size)] for y in range(board_size)]
    shallow_slots[next_slot.row][next_slot.col] = 1
    dist_from_edge = 0

    while dist_from_edge < board_size / 2:
        if next_slot.row + dist_from_edge < board_size - 1 and next_slot.col == dist_from_edge :
            direction = 's'
        elif next_slot.row + dist_from_edge == board_size - 1 and next_slot.col + dist_from_edge < board_size - 1:
            direction = 'e'
        elif next_slot.row > dist_from_edge and next_slot.col + dist_from_edge == board_size - 1:
            direction = 'n'
        elif next_slot.row == dist_from_edge and next_slot.col + dist_from_edge >= board_size - 1 :
            direction = 'w'

        if direction == 's':
            new_slot = next_slot.go_south()
        elif direction == 'e':
            new_slot = next_slot.go_east()
        elif direction == 'n':
            new_slot = next_slot.go_north()
        elif direction == 'w':
            new_slot  = next_slot.go_west()

        if shallow_slots[new_slot.row][new_slot.col] == 1:
            dist_from_edge += 1
            continue
        else:
            next_slot = new_slot

        steps.append(next_slot)
        shallow_slots[next_slot.row][next_slot.col] = 1

    return steps


def get_spiraling_out_steps(agent, board_size):
    """
    This function return the agent steps, when deciding to cover the world, spiraling from inside to outside.
    Note: this coverage method is not optimal!
    :param agent: the agent covering the world
    :param board_size: self explanatory
    :return: list of steps
    """
    steps = []

    next_slot = Slot(agent.InitPosX, agent.InitPosY)
    # next_slot = Slot(36,6)

    # start by going toward the center
    if next_slot.row < board_size / 2:
        while next_slot.row < board_size / 2:
            next_slot = next_slot.go_south()
            steps.append(next_slot)
            continue
    else:
        while next_slot.row > board_size / 2:
            next_slot = next_slot.go_north()
            steps.append(next_slot)
            continue

    if next_slot.col < board_size / 2:
        while next_slot.col < board_size / 2:
            next_slot = next_slot.go_east()
            steps.append(next_slot)
            continue
    else:
        while next_slot.col > board_size / 2:
            next_slot = next_slot.go_west()
            steps.append(next_slot)
            continue
    # steps.append(next_slot)

    # after reaching the center, start covering, counter clockwise
    circ = 0
    counter = 1
    while circ < board_size:
        circ += 1
        if circ < board_size:
            for _ in range(circ):
                next_slot = next_slot.go_west()
                steps.append(next_slot)
                counter += 1
        if circ < board_size:
            for _ in range(circ):
                next_slot = next_slot.go_north()
                steps.append(next_slot)
                counter += 1

        circ += 1
        if circ < board_size:
            for _ in range(circ):
                next_slot = next_slot.go_east()
                steps.append(next_slot)
                counter += 1
        if circ < board_size:
            for _ in range(circ):
                next_slot = next_slot.go_south()
                steps.append(next_slot)
                counter += 1

    for step in steps:
        if step.row > 49 or step.col > 49 or step.row < 0 or step.col < 0:
            print steps
            break
    return steps


def run_agent_over_board_full_knowledge_interception_strategy(agent_r, agent_o, board_size_x, board_size_y):
    steps_o = []

    if agent_o.Strategy == StrategyEnum.VerticalCoverageCircular:
        steps_o = get_vertical_coverage_steps(agent_o, board_size_x, board_size_y)
    elif agent_o.Strategy == StrategyEnum.HorizontalCoverageCircular:
        steps_o = get_horizontal_coverage_steps(agent_o, board_size_x, board_size_y)
    elif agent_o.Strategy == StrategyEnum.QuartersCoverageCircular:
        steps_o = get_quarters_coverage_steps(agent_o, board_size_x, board_size_y)

    # Find interception point
    (interceptionPoint_R_O, distance) = get_interception_point(steps_o, agent_r.InitPosX, agent_r.InitPosY)

    steps_r = run_agent_over_board_interception_strategy(steps_o, agent_r.InitPosX, agent_r.InitPosY,
                                                         interceptionPoint_R_O)

    # play steps for both players and for each step check who covered it first

    return steps_r


def go_from_a_to_b(a, b):
    """
    Returns a list of steps from A to B
    :param a: First Slot
    :param b: Second Slot
    :return: list of steps from A to B
    """

    current_step = a
    steps_to_return = [current_step]

    while not current_step.row == b.row:
        current_step = current_step.go_north() if b.row < a.row else current_step.go_south()
        steps_to_return.append(current_step)

    while not current_step.col == b.col:
        current_step = current_step.go_east() if b.col > a.col else current_step.go_west()
        steps_to_return.append(current_step)

    return steps_to_return


def get_farthest_corner(a, board_size):
    """
    return the farthest corner from a given position
    :param a: the given position
    :param board_size: the size of the given game board
    :return: the farthest corner from A
    """
    f_row = 0 if a.row < board_size / 2 else board_size - 1
    f_col = 0 if a.col < board_size / 2 else board_size - 1
    return Slot(f_row, f_col)


def get_vertical_coverage_farthest_corner_opponent_aware_steps(self, board_size, agent_o):
    """
    This function returns the coverage steps,when covering knowing io, and start covering vertically from the farthest
    corner.
    :param self:
    :param board_size:
    :param agent_o:
    :return: the coverage steps.
    """
    assert agent_o is not None
    steps = []

    # go to the farthest corner
    steps.extend(go_from_a_to_b(a=Slot(self.InitPosX, self.InitPosY),
                                b=get_farthest_corner(Slot(agent_o.InitPosX, agent_o.InitPosY), board_size=board_size)))

    # from there, cover vertically
    current_slot = steps[-1]
    v_dir = 'u' if current_slot.row == board_size - 1 else 'd'
    h_dir = 'l' if current_slot.col == board_size - 1 else 'r'
    counter = 1
    while counter <= board_size*board_size:
        if v_dir == 'u':
            while current_slot.row > 0:
                current_slot = current_slot.go_north()
                steps.append(current_slot)
                counter += 1

            if counter == board_size*board_size:
                break

            current_slot = current_slot.go_west() if h_dir == 'l' else current_slot.go_east()
            v_dir = 'd'
            steps.append(current_slot)
            counter += 1
            continue
        else:
            while current_slot.row < board_size - 1:
                current_slot = current_slot.go_south()
                steps.append(current_slot)
                counter += 1

            if counter == board_size*board_size:
                break

            current_slot = current_slot.go_west() if h_dir == 'l' else current_slot.go_east()
            v_dir = 'u'
            steps.append(current_slot)
            counter += 1
            continue

    return steps


def get_semi_cycle_coverage_farthest_corner_opponent_aware_steps(self, board_size, agent_o):
    """
    This function returns the coverage steps, when covering knowing io, starting from the farthest corner from io, and
    covering semi-cyclic - covering the closer layers first.
    :param self:
    :param board_size:
    :param agent_o:
    :return: the coverage steps.
    """
    assert agent_o is not None
    steps = []

    # go to the farthest corner
    steps.extend(go_from_a_to_b(a=Slot(self.InitPosX, self.InitPosY),
                                b=get_farthest_corner(Slot(agent_o.InitPosX, agent_o.InitPosY), board_size=board_size)))

    # from there, cover semi-cyclic
    current_slot = steps[-1]
    v_dir = 'u' if current_slot.row == board_size - 1 else 'd'
    h_dir = 'r' if current_slot.col == board_size - 1 else 'l'
    start_vertical = True
    distance = 1
    counter = 1

    # initial horizontal step
    current_slot = current_slot.go_west() if h_dir == 'r' else current_slot.go_east()
    steps.append(current_slot)
    counter += 1

    while counter <= board_size*board_size:
        if start_vertical:
            # going vertically
            for _ in xrange(distance):
                current_slot = current_slot.go_north() if v_dir == 'u' else current_slot.go_south()
                steps.append(current_slot)
                counter += 1

            # going horizontally
            for _ in xrange(distance):
                current_slot = current_slot.go_west() if h_dir == 'l' else current_slot.go_east()
                steps.append(current_slot)
                counter += 1

            # final vertical step
            if counter < board_size * board_size:
                current_slot = current_slot.go_north() if v_dir == 'u' else current_slot.go_south()
                steps.append(current_slot)
                counter += 1

        else:
            # going horizontally
            for _ in xrange(distance):
                current_slot = current_slot.go_west() if h_dir == 'l' else current_slot.go_east()
                steps.append(current_slot)
                counter += 1

            # going vertically
            for _ in xrange(distance):
                current_slot = current_slot.go_north() if v_dir == 'u' else current_slot.go_south()
                steps.append(current_slot)
                counter += 1

            # final horizontal step
            if counter < board_size*board_size:
                current_slot = current_slot.go_west() if h_dir == 'l' else current_slot.go_east()
                steps.append(current_slot)
                counter += 1

        start_vertical = not start_vertical
        h_dir = 'r' if h_dir == 'l' else 'l'
        v_dir = 'u' if v_dir == 'd' else 'd'
        distance += 1

    return steps


def get_spiraling_out_from_opp_opponent_aware(self, board_size, agent_o):
    """
    This function returns the coverage steps, choosing to start from the initial position of the opponent, io, and from
    there cover the world circling out.
    For convenience purposes, we rounding to the closest even position, and from there covering as explained above.
    :param self: the agent
    :param board_size:
    :param agent_o:
    :return:
    """
    #TODO: create strategy class. Create strategies for all the other options, and put the relevant methods inside each item
    steps = []

    # Make assertions over opp initial position: we want it to be with even column and odd row.
    # Then, go to the fixed initial position
    fixed_io = Slot(agent_o.InitPosX, agent_o.InitPosY)
    if fixed_io.row % 2 == 0:
        fixed_io.row += 1
    if fixed_io.col % 2 != 0:
        fixed_io.col -= 1
    steps.extend(go_from_a_to_b(Slot(self.InitPosX, self.InitPosY), fixed_io))

    # cover the world, circling from this position out. Taking big steps to one direction to compensate for the other.
    h_dir = 'l' if fixed_io.col > board_size / 2 else 'r'
    v_dir = 'u' if fixed_io.row > board_size / 2 else 'd'

    v_step_size = (board_size - fixed_io.row - 1) / (fixed_io.row + 1)
    h_step_size = (board_size - fixed_io.col - 1) / (fixed_io.col + 1)

    current_slot = steps[-1]

    while True:
        # going horizontally long
        for _ in h_step_size:
            current_slot.go(h_dir)
            steps.append(current_slot)

        current_slot = current_slot.go(v_dir)
        steps.append(current_slot)

        # going horizontally back, covering vertically
        for _ in xrange(h_step_size+2):
            for i in xrange(v_step_size):
                current_slot = current_slot.go(v_dir, opposite_direction=False if i % 2 == 0 else True)
                steps.append(current_slot)

            current_slot = current_slot.go(h_dir, opposite_direction=True)
            steps.append(current_slot)


        # going vertically back
        #todo: doesn't cover all the cases! finish method!

