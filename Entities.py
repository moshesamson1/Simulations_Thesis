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


class Slot:
    def __init__(self, x, y):
        self.has_been_visited = False
        self.is_occupied = False
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

    def __str__(self):
        return "{0},{1},{2}".format(str(int(self.row)), str(int(self.col)), str(self.covered_by))

    def __repr__(self):
        return str(self)


class StrategyEnum:
    def __init__(self):
        pass

    VerticalCoverageCircular, HorizontalCoverageCircular, FullKnowledgeInterceptionCircular, QuartersCoverageCircular,\
        RandomSTC = range(5)


class Agent:
    def __init__(self, name, strategy, x, y, board=None, agent_o=None):
        # type: (str, StrategyEnum, int, int, Board, Agent) -> None
        assert isinstance(strategy, StrategyEnum)

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


class Game:

    def __init__(self, board, agent_r, agent_o):
        self._board = board
        self._agentR = agent_r
        self._agentO = agent_o

    def run_game(self):
        steps_r = self._agentR.steps
        steps_o = self._agentO.steps

        # print steps_o

        if len(steps_o) == len(steps_r):
            for i in range(len(steps_r)):
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
        else:
            print "what?!"

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

    # print "init pos {},{}: ".format(agent.InitPosX, agent.InitPosY)

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


def get_quarters_coverage_steps(agent, board_size_x, board_size_y):
    # This coverage strategy covers first the top left, then top right, then bottom left, then bottom right quarters of
    # the area.
    # While building this function, we assumed 100X100 dimensions

    next_slot = Slot(agent.InitPosX, agent.InitPosY)
    # flag = (agent.InitPosY == boardSizeY - 1 and not (agent.InitPosX == 0))

    steps = []
    counter = 0

    # next_slot = Slot(10,6)
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
