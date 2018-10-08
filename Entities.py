from abc import ABCMeta
from abc import abstractmethod
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from os.path import basename
import smtplib


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
        SemiCyclingFromFarthestCorner_OpponentAware, CircleOutsideFromIo = range(11)


class Agent:
    def __init__(self, name, strategy_enum, x, y, board=None, agent_o=None):
        # type: (str, int, int, int, Board, Agent) -> None
        assert isinstance(strategy_enum, int)

        self.Name = name
        self.StrategyEnum = strategy_enum
        self.InitPosX = x
        self.InitPosY = y
        self.gameBoard = board

        self.Strategy = Strategy.get_strategy_from_enum(strategy_enum)
        self.steps = self.Strategy.get_steps(self, len(board.Slots), agent_o)


class Game:

    def __init__(self, board, agent_r, agent_o):
        self._board = board
        self._agentR = agent_r
        self._agentO = agent_o

    def run_game(self, optimality=True):
        steps_r = self._agentR.steps
        steps_o = self._agentO.steps


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

class Strategy:
    __metaclass__ = ABCMeta
    steps = None  # type: List[Any]
    def __init__(self):
        self.steps = []

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
        f_row = 0 if a.row < board_size / 2 else board_size - 1
        f_col = 0 if a.col < board_size / 2 else board_size - 1
        return Slot(f_row, f_col)

    @classmethod
    def get_strategy_from_enum(cls, strategy_enum):
        # type: (Strategy, int) -> Strategy
        from Strategies import *
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
        elif strategy_enum == StrategyEnum.CircleOutsideFromIo:
            return CircleOutsideFromIo_Strategy.CircleOutsideFromIo_Strategy()


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
