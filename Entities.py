import smtplib
from abc import ABCMeta
from abc import abstractmethod
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from enum import Enum
from os.path import basename

import matplotlib.pyplot as plt
import numpy as np


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


class StrategyEnum(Enum):
    VerticalCoverageCircular = 0
    HorizontalCoverageCircular = 1
    FullKnowledgeInterceptionCircular = 2
    QuartersCoverageCircular = 3
    RandomSTC = 4
    VerticalCoverageNonCircular = 5
    SpiralingOut = 6
    SpiralingIn = 7
    VerticalFromFarthestCorner_OpponentAware = 8
    SemiCyclingFromFarthestCorner_OpponentAware = 9
    SemiCyclingFromAdjacentCorner_row_OpponentAware = 10
    SemiCyclingFromAdjacentCorner_col_OpponentAware = 11
    CircleOutsideFromIo = 12,
    GrowingDiagonalsCornerToCorner = 13


class Agent:
    def __init__(self, name: str, strategy_enum: StrategyEnum, x: int, y: int, board: Board = None,
                 agent_o: object = None) -> None:

        assert isinstance(strategy_enum, Enum)

        self.Name = name
        self.StrategyEnum = strategy_enum
        self.InitPosX = x
        self.InitPosY = y
        self.gameBoard = board

        self.Strategy = Strategy.get_strategy_from_enum(strategy_enum)
        self.steps = self.Strategy.get_steps(self, len(board.Slots), agent_o)

    def get_interception_time_of_slot(self, slot: Slot):
        assert len(self.steps) >= 0
        return self.steps.index(slot)

    def get_strategy(self):
        return self.Strategy.__str__()

    def get_strategy_short(self):
        return self.get_strategy()[:5] + "..."

    def display_heat_map(self,x,y):
        arr = self.get_heatmap()
        DisplayingClass.create_heat_map(arr, x, y, self.get_strategy_short())

    def get_heatmap(self):
        arr = np.zeros((self.gameBoard.Rows, self.gameBoard.Cols))
        for id in range(len(self.steps)):
            if arr[self.steps[id].row][self.steps[id].col] == 0:
                arr[self.steps[id].row][self.steps[id].col] = id
        return arr

    def get_cross_heatmap(self, other, probabilities=[0.5, 0.5]):
        my_hm = probabilities[0] * self.get_heatmap()
        o_hm = probabilities[1] * other.get_heatmap()
        return np.add(my_hm, o_hm)

    def get_sub_heatmap(self, other_hm, probabilities=[0.5, 0.5]):
        my_hm = probabilities[0] * self.get_heatmap()
        o_hm = probabilities[1] * other_hm
        return np.subtract(my_hm, o_hm)

    def display_cross_heatmap(self, other, display_grid_x, display_grid_y, probabilities):
        c = self.get_cross_heatmap(other, probabilities)
        DisplayingClass.create_heat_map(c, display_grid_x, display_grid_y,
                                        "comb. of \n({0} and \n{1}):".format(
                                            str(self.get_strategy_short()), str(other.get_strategy_short())))
        return c

    def display_sub_heatmap(self, other_hm, display_grid_x, display_grid_y, probabilities):
        c = self.get_sub_heatmap(other_hm, probabilities)

        DisplayingClass.create_heat_map(c, display_grid_x, display_grid_y,
                                        "subs. sum: {}\navg: {}\n std: {}".format(np.sum(c),
                                                                                  np.average(c),
                                                                                  np.std(c)))

class Game:
    def __init__(self, agent_r: Agent, agent_o: Agent, size=(100,100)) -> None:
        self._board = Board(size[0], size[1])
        self._agentR = agent_r
        self._agentO = agent_o

    def run_game(self, enforce_paths_length=False):
        steps_r = self._agentR.steps
        steps_o = self._agentO.steps

        if enforce_paths_length:
            if not len(steps_o) == len(steps_r):
                raise AssertionError("wrong length! len(steps_o)={}, len(steps_r)={}".format(len(steps_o),len(steps_r)))

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

        return self.get_r_gain(), self.get_o_gain()
    def get_r_gain(self):
        cond_count = 0

        # print self._board.Slots

        for i in range(0, self._board.Rows):
            for j in range(0, self._board.Cols):
                if self._board.Slots[i][j].covered_by == self._agentR.Name:
                    cond_count += 1

        return float(cond_count)

    def get_o_gain(self):
        cond_count = 0

        size_x = len(self._board.Slots)
        size_y = len(self._board.Slots[0])

        for i in range(0, size_x):
            for j in range(0, size_y):
                if self._board.Slots[i][j].covered_by == self._agentO.Name:
                    cond_count += 1

        return float(cond_count)

    @property
    def board(self):
        return self._board


class Strategy:
    __metaclass__ = ABCMeta
    steps = [] # type: List[Slot]

    def __init__(self):
        self.steps = []
        self.set_steps = set()

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    @abstractmethod
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        """Returns the steps agent perform to cover the world"""

    @classmethod
    def go_from_a_to_b(self, a, b):
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

    @classmethod
    def get_farthest_corner(self, a, board_size):
        """
        return the farthest corner from a given position
        :param a: the given position
        :param board_size: the size of the given game board
        :return: the farthest corner from A
        """
        f_row = 0 if a.row > board_size / 2 else board_size - 1
        f_col = 0 if a.col > board_size / 2 else board_size - 1
        return Slot(f_row, f_col)

    @classmethod
    def get_adjacent_corner(self, a, board_size, first_option):
        """

        :param a:
        :param board_size:
        :return:
        """
        if first_option:
            f_row = 0 if a.row < board_size / 2 else board_size - 1
            f_col = 0 if a.col > board_size / 2 else board_size - 1
        else:
            f_row = 0 if a.row > board_size / 2 else board_size - 1
            f_col = 0 if a.col < board_size / 2 else board_size - 1
        return Slot(f_row, f_col)

    @classmethod
    def get_strategy_from_enum(cls, strategy_enum: StrategyEnum):
        from Strategies import VerticalCircularCoverage_Strategy, HorizontalCircularCoverage_Strategy, \
            InterceptThenCopy_Strategy, CoverByQuarters_Strategy, STC_Strategy,VerticalNonCircularCoverage_Strategy,\
            CircleInsideFromCornerFarthestFromIo_Strategy, CircleOutsideFromBoardCenter_Strategy,\
            VerticalCoverageFromCornerFarthestFromIo_Strategy, CircleOutsideFromCornerFarthestFromIo_Strategy, \
            CircleOutsideFromIo_Strategy, CircleOutsideFromCornerAdjacentToIo_Strategy, \
            GrowingDiagonalsCornerToCorner_Strategy

        if strategy_enum == StrategyEnum.VerticalCoverageCircular:
            return VerticalCircularCoverage_Strategy.VerticalCircularCoverage_Strategy()
        elif strategy_enum == StrategyEnum.HorizontalCoverageCircular:
            return HorizontalCircularCoverage_Strategy.HorizontalCircularCoverage_Strategy()
        elif strategy_enum == StrategyEnum.FullKnowledgeInterceptionCircular:
            return InterceptThenCopy_Strategy.InterceptThenCopy_Strategy()
        elif strategy_enum == StrategyEnum.QuartersCoverageCircular:
            return CoverByQuarters_Strategy.CoverByQuarters_Strategy()
        elif strategy_enum == StrategyEnum.RandomSTC:
            return STC_Strategy.STC_Strategy()
        elif strategy_enum == StrategyEnum.VerticalCoverageNonCircular:
            return VerticalNonCircularCoverage_Strategy.VerticalNonCircularCoverage_Strategy()
        elif strategy_enum == StrategyEnum.SpiralingIn:
            return CircleInsideFromCornerFarthestFromIo_Strategy.CircleInsideFromCornerFarthestFromIo_Strategy()
        elif strategy_enum == StrategyEnum.SpiralingOut:
            return CircleOutsideFromBoardCenter_Strategy.CircleOutsideFromBoardCenter_Strategy()
        elif strategy_enum == StrategyEnum.VerticalFromFarthestCorner_OpponentAware:
            return VerticalCoverageFromCornerFarthestFromIo_Strategy.VerticalCoverageFromCornerFarthestFromIo_Strategy()
        elif strategy_enum == StrategyEnum.SemiCyclingFromFarthestCorner_OpponentAware:
            return CircleOutsideFromCornerFarthestFromIo_Strategy.CircleOutsideFromCornerFarthestFromIo_Strategy()
        elif strategy_enum == StrategyEnum.SemiCyclingFromAdjacentCorner_col_OpponentAware:
            return CircleOutsideFromCornerAdjacentToIo_Strategy.CircleOutsideFromCornerAdjacentToIo_Strategy(False)
        elif strategy_enum == StrategyEnum.SemiCyclingFromAdjacentCorner_row_OpponentAware:
            return CircleOutsideFromCornerAdjacentToIo_Strategy.CircleOutsideFromCornerAdjacentToIo_Strategy(True)
        elif strategy_enum == StrategyEnum.CircleOutsideFromIo:
            return CircleOutsideFromIo_Strategy.CircleOutsideFromIo_Strategy()
        elif strategy_enum == StrategyEnum.GrowingDiagonalsCornerToCorner:
            return GrowingDiagonalsCornerToCorner_Strategy.GrowingDiagonalsCornerToCorner_Strategy()

    def add_step(self, step):
        # if step not in self.set_steps:
        self.steps.append(step)
        self.set_steps.add(step)


def send_files_via_email(text, title, file_name):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    password = input("Password for moshe.samson@mail.huji.ac.il: ")
    server.login("moshe.samson@mail.huji.ac.il", password)

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


class DisplayingClass:
    fig, ax = plt.subplots(5, 3)

    @staticmethod
    def get_plt():
        return plt

    @staticmethod
    def create_heat_map(arr, x,y, title=''):
        # arr = np.array(array)

        DisplayingClass.ax[x][y].imshow(arr)
        DisplayingClass.ax[x][y].set_title(title)
        DisplayingClass.ax[x][y].set_yticklabels([])
        DisplayingClass.ax[x][y].set_xticklabels([])
        # DisplayingClass.fig.tight_layout()

        # Create colorbar
        # cbar = DisplayingClass.ax[x][y].figure.colorbar(im, ax=DisplayingClass.ax[x][y])
        # cbar.ax.set_ylabel("color bar label", rotation=-90, va="bottom")

        return plt


