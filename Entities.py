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
        self.HasBeenVisited = False
        self.CurrentlyOccupied = False
        self.BeenFirstCoveredBy = "*"
        self.row = x
        self.col = y
        self.Name = "("+str(x) + "," + str(y) + ")"

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __ne__(self, other):
        return self.row != other.row or self.col != other.col

    def __hash__(self):
        return hash((self.row, self.col))

    def GoLeft(self):
        return Slot(self.row, self.col - 1)
    def GoRight(self):
        return Slot(self.row, self.col + 1)
    def GoUp(self):
        return Slot(self.row - 1, self.col)
    def GoDown(self):
        return Slot(self.row + 1, self.col)



    def __str__(self):
        return str(int(self.row)) + "," +str(int(self.col)) + "," + str(self.BeenFirstCoveredBy)

    def __repr__(self):
        return str(self)


class Agent:
    def __init__(self, name, Strategy, x, y, board, agent_0=None):
        # type: (string, string, int, int, Board, object) -> object
        self.Name = name
        self.Strategy = Strategy
        self.InitPosX = x
        self.InitPosY = y
        self.gameBoard = board

        if self.Strategy == "VerticalCoverage":
            self.steps = GetVerticalCoverageSteps(self, len(self.gameBoard.Slots), len(self.gameBoard.Slots[0]))
        elif self.Strategy == "HorizontalCoverage":
            self.steps = GetHorizontalCoverageSteps(self, len(self.gameBoard.Slots), len(self.gameBoard.Slots[0]))
        elif self.Strategy == "FullKnowledgeInterception":
            self.steps = RunAgentOverBoardFullKnowledgeInterceptionStrategy(self, agent_0,
                                                                        len(self.gameBoard.Slots),
                                                                        len(self.gameBoard.Slots[0]))
        elif self.Strategy == "QuartersCoverage":
            self.steps = GetQuartersCoverageSteps(self, len(self.gameBoard.Slots), len(self.gameBoard.Slots[0]))
        elif self.Strategy == "random":
            self.steps = SpanningTreeCoverage.get_random_coverage_strategy(len(self.gameBoard.Slots), Slot(self.InitPosX, self.InitPosY), print_mst=False)


class Game:

    def __init__(self, board, agentR, agentO):
        self._board = board
        self._agentR = agentR
        self._agentO = agentO

    def RunGame(self):
        stepsR = self._agentR.steps
        stepsO = self._agentO.steps

        # print stepsO

        if len(stepsO) == len(stepsR):
            for i in range(len(stepsR)):
                # perform step for R
                stepR = stepsR[i]
                if not self._board.Slots[int(stepR.row)][int(stepR.col)].HasBeenVisited:
                    self._board.Slots[int(stepR.row)][int(stepR.col)].HasBeenVisited = True
                    self._board.Slots[int(stepR.row)][int(stepR.col)].BeenFirstCoveredBy = self._agentR.Name

                # then perform step for O
                stepO = stepsO[i]
                if not self._board.Slots[int(stepO.row)][int(stepO.col)].HasBeenVisited:
                    self._board.Slots[int(stepO.row)][int(stepO.col)].HasBeenVisited = True
                    self._board.Slots[int(stepO.row)][int(stepO.col)].BeenFirstCoveredBy = self._agentO.Name
        else:
            print "what?!"

    def GetRGain(self):
        condCount = 0

        # print self._board.Slots

        for i in xrange(0,self._board.Rows):
            for j in xrange(0,self._board.Cols):
                if self._board.Slots[i][j].BeenFirstCoveredBy == self._agentR.Name:
                    condCount += 1

        return float(condCount)

    def GetOGain(self):
        condCount = 0

        size_x = len(self._board.Slots)
        size_y = len(self._board.Slots[0])

        for i in xrange(0, size_x):
            for j in xrange(0, size_y):
                if self._board.Slots[i][j].BeenFirstCoveredBy == self._agentO.Name:
                    condCount += 1

        return float(condCount)


def RunAgentOverBoardInterceptionSrtategy(stepsO, R_initX, R_initY, interceptionPoint):
    distance2InterceptionPoint = fabs(R_initX -interceptionPoint[0]) + fabs(R_initY - interceptionPoint[1])
    nextSlot = (R_initX, R_initY)
    steps = []
    steps.append(nextSlot)
    counter = 0

    if np.sign(interceptionPoint[0] - R_initX) >= 0:
        xSign = 1
    else:
        xSign = -1

    if np.sign(interceptionPoint[1] - R_initY) >= 0:
        ySign = 1
    else:
        ySign = -1

    while True:
        if counter >= len(stepsO)-1: break

        # first, go to interception point
        if counter < distance2InterceptionPoint:
            for x_step in xrange(R_initX + xSign, interceptionPoint[0] + xSign, xSign):
                counter += 1
                nextSlot = (x_step, nextSlot[1])
                steps.append(nextSlot)
            for y_step in xrange(R_initY + ySign, interceptionPoint[1] + ySign, ySign):
                counter += 1
                nextSlot = (nextSlot[0], y_step)
                steps.append(nextSlot)
        # then, play as O plays
        else:
            counter += 1
            nextSlot = stepsO[counter]
            steps.append(nextSlot)

    return steps


def GetInterceptionPoint(stepsO, R_initX, R_initY):
    ip_x = -1
    ip_y = -1

    stepsCounter = 0
    for step in stepsO:
        stepsCounter += 1
        distanceFromRInitPos = fabs(step[0] - R_initX) + fabs(step[1] - R_initY) + 1
        if fabs(stepsCounter - distanceFromRInitPos) <= 1:
            ip_x, ip_y = step
            break

    assert (ip_x != -1 and ip_y != -1)

    return (ip_x, ip_y), fabs(ip_x - R_initX) + fabs(ip_y - R_initY)


def GetVerticalCoverageSteps(agent, boardSizeX, boardSizeY):
    nextSlot = (agent.InitPosX, agent.InitPosY)

    flag = (agent.InitPosY == boardSizeY - 1 and not (agent.InitPosX == 0))

    steps = []
    counter = 0

    # print "init pos {},{}: ".format(agent.InitPosX, agent.InitPosY)

    while True:
        counter += 1
        if counter > 1000000000000:
            break

        steps.append(nextSlot)
        # in the middle of moving from bottom row to top row
        if flag:
            nextSlot = (nextSlot[0] - 1, nextSlot[1])
            if nextSlot[0] == 0:
                flag = False
            if IsNextSlotTheInitialPos(nextSlot, (agent.InitPosX, agent.InitPosY)):
                break
            continue
        # check if in last position, and start moving from last row to top row
        elif nextSlot[0] == boardSizeX - 1 and nextSlot[1] == boardSizeY - 1 - 1:
            flag = True
            nextSlot = (nextSlot[0], nextSlot[1] + 1)
            if IsNextSlotTheInitialPos(nextSlot, (agent.InitPosX, agent.InitPosY)):
                break
            continue
        # update next slot
        elif nextSlot[0] % 2 != 0:
            if nextSlot[1] == boardSizeY - 1 - 1:
                nextSlot = (nextSlot[0] + 1, nextSlot[1])
            else:
                nextSlot = (nextSlot[0], nextSlot[1] + 1)
        else:
            if nextSlot[1] == 0:
                nextSlot = (nextSlot[0] + 1, nextSlot[1])
            else:
                nextSlot = (nextSlot[0], nextSlot[1] - 1)

        if IsNextSlotTheInitialPos(nextSlot, (agent.InitPosX, agent.InitPosY)):
            break

    return steps


def GetQuartersCoverageSteps(agent, boardSizeX, boardSizeY):
    # This coverage strategy covers first the top left, then top right, then bottom left, then bottom right quarters of
    # the area.
    # While building this function, we assumed 100X100 dimensions

    next_slot = Slot(agent.InitPosX, agent.InitPosY)
    #flag = (agent.InitPosY == boardSizeY - 1 and not (agent.InitPosX == 0))

    steps = []
    counter = 0

    #next_slot = Slot(10,6)
    while True:
        counter+=1

        if counter > 100000:
            print "Something is wrong, counter is too big!"
            print steps
            break


        if counter > 1 and IsNextSlotTheInitialPos((next_slot.row, next_slot.col), (agent.InitPosX, agent.InitPosY)):
            break

        steps.append((next_slot.row, next_slot.col))

        # TL Quarter
        if 0 <= next_slot.row < boardSizeX/2 and 0 <= next_slot.col < boardSizeY/2:
            if (next_slot.row, next_slot.col) == (boardSizeX/2-1,boardSizeY/2-1):
                next_slot.row, next_slot.col = next_slot.GoRight()
                continue
            if next_slot.col == 0:
                next_slot.row, next_slot.col = next_slot.GoUp() if next_slot.row > 0 else next_slot.GoRight()
                continue
            if next_slot.row % 2 == 0 and next_slot.row < boardSizeX/2-2: # An even line, not the last one
                next_slot.row, next_slot.col = next_slot.GoRight() if not next_slot.col == boardSizeY / 2 - 1 else next_slot.GoDown()
                continue
            elif next_slot.row % 2 != 0 and not next_slot.row == boardSizeX/2-1: # An odd line, not before last
                next_slot.row, next_slot.col = next_slot.GoLeft() if not next_slot.col == 1 else next_slot.GoDown()
                continue
            elif next_slot.row % 2 == 0 and next_slot.row == boardSizeX/2-2: # An even line, the last one
                next_slot.row, next_slot.col = next_slot.GoDown() if next_slot.col % 2 != 0 else next_slot.GoRight()
            elif next_slot.row % 2 != 0 and next_slot.row == boardSizeX/2-1:  # An odd line, last line
                next_slot.row, next_slot.col = next_slot.GoRight() if next_slot.col % 2 != 0 else next_slot.GoUp()
            else:
                print "TL: Error occurred! Should not reach here!"
            continue

        # TR Quarter
        elif 0 <= next_slot.row < boardSizeX/2 and boardSizeY/2 <= next_slot.col < boardSizeY:
            if (next_slot.row,next_slot.col) == (boardSizeX/2-1,boardSizeY-1):
                next_slot.row,next_slot.col = next_slot.GoDown()
                continue
            elif next_slot.col % 2 == 0:
                next_slot.row, next_slot.col = next_slot.GoUp() if next_slot.row > 0 else next_slot.GoRight()
                continue
            elif next_slot.col % 2 != 0:
                next_slot.row, next_slot.col = next_slot.GoDown() if next_slot.row < boardSizeX/2-1 else next_slot.GoRight()
                continue
            else:
                print "TR: Error occurred! Should not reach here!"
            continue

        # BL Quarter
        elif boardSizeX/2 <= next_slot.row < boardSizeX and 0 <= next_slot.col < boardSizeY/2:
            if (next_slot.row, next_slot.col) == (boardSizeX/2, 0): # last cell of quarter
                next_slot.row, next_slot.col= next_slot.GoUp()
                continue
            elif next_slot.col % 2 == 0: # an even column
                next_slot.row, next_slot.col = next_slot.GoUp() if next_slot.row > boardSizeX / 2 else next_slot.GoLeft()
                continue
            elif next_slot.col % 2 != 0:  # An odd line
                next_slot.row, next_slot.col = next_slot.GoDown() if next_slot.row < boardSizeX - 1 else next_slot.GoLeft()
                continue
            else:
                print "BL: Error occurred! Should not reach here!"
            continue

        # BR Quarter
        else:
            if (next_slot.row, next_slot.col) == (boardSizeX/2, boardSizeY/2):  # last cell of quarter
                next_slot.row, next_slot.col = next_slot.GoLeft()
                continue
            elif next_slot.col == boardSizeY-1:
                next_slot.row, next_slot.col = next_slot.GoDown() if not next_slot.row == boardSizeX-1 else next_slot.GoLeft()
                continue
            elif next_slot.row % 2 != 0 and not next_slot.row == boardSizeX/2+1: # and odd line, not before last
                next_slot.row, next_slot.col = next_slot.GoLeft() if next_slot.col > boardSizeY/2 else next_slot.GoUp()
                continue
            elif next_slot.row % 2 != 0 and next_slot.row == boardSizeX/2+1: # and odd line, DO before last
                next_slot.row, next_slot.col = next_slot.GoUp() if next_slot.col % 2 == 0 else next_slot.GoLeft()
                continue
            elif next_slot.row % 2 == 0 and not next_slot.row == boardSizeX/2:  # an even line, not last one
                next_slot.row, next_slot.col = next_slot.GoRight() if next_slot.col < boardSizeY - 2 else next_slot.GoUp()
                continue
            elif next_slot.row % 2 == 0 and next_slot.row == boardSizeX/2:  # an even line, INDEED last one
                next_slot.row, next_slot.col = next_slot.GoLeft() if next_slot.col % 2 == 0 else next_slot.GoDown()
                continue
            else:
                print "BR: Error occurred! Should not reach here!"
            continue

    return steps


def GetHorizontalCoverageSteps(agent, boardSizeX, boardSizeY):
    nextSlot = (agent.InitPosX, agent.InitPosY)
    flag = (agent.InitPosX == boardSizeX - 1)

    steps = []
    counter = 0

    # print "init pos {},{}: ".format(agent.InitPosX, agent.InitPosY)

    while True:
        counter += 1
        if counter > boardSizeX * boardSizeY:
            break

        steps.append(nextSlot)
        # in the middle of moving from bottom rpw to top row
        if flag:
            if nextSlot[1] == boardSizeY - 1:
                flag = False
                nextSlot = (nextSlot[0] - 1, nextSlot[1])
            else:
                nextSlot = (nextSlot[0], nextSlot[1] + 1)
            if IsNextSlotTheInitialPos(nextSlot, (agent.InitPosX, agent.InitPosY)):
                break
            continue
        # check if in last position, and start moving from last row to top row
        elif nextSlot[0] == boardSizeX - 1 - 1 and nextSlot[1] == 0:
            flag = True
            nextSlot = (nextSlot[0] + 1, nextSlot[1])

            if IsNextSlotTheInitialPos(nextSlot, (agent.InitPosX, agent.InitPosY)):
                break
            continue
        # update next slot
        elif nextSlot[1] % 2 != 0:
            if nextSlot[0] == 0:
                nextSlot = (nextSlot[0], nextSlot[1] - 1)
            else:
                nextSlot = (nextSlot[0] - 1, nextSlot[1])
        else:
            if nextSlot[0] == boardSizeY - 1 - 1:
                nextSlot = (nextSlot[0], nextSlot[1] - 1)
            else:
                nextSlot = (nextSlot[0] + 1, nextSlot[1])
                if nextSlot[0] > boardSizeX - 1:
                    print "+1"

        if IsNextSlotTheInitialPos(nextSlot, (agent.InitPosX, agent.InitPosY)):
            break

    return steps


def RunAgentOverBoardFullKnowledgeInterceptionStrategy(agentR, agentO, boardSizeX, boardSizeY):
    stepsO = []

    if agentO.Strategy == "VerticalCoverage":
        stepsO = GetVerticalCoverageSteps(agentO, boardSizeX, boardSizeY)
    elif agentO.Strategy == "HorizontalCoverage":
        stepsO = GetHorizontalCoverageSteps(agentO, boardSizeX, boardSizeY)
    elif agentO.Strategy == "QuartersCoverage":
        stepsO = GetQuartersCoverageSteps(agentO, boardSizeX, boardSizeY)

    # Find interception point
    (interceptionPoint_R_O, distance) = GetInterceptionPoint(stepsO, agentR.InitPosX, agentR.InitPosY)

    stepsR = RunAgentOverBoardInterceptionSrtategy(stepsO, agentR.InitPosX, agentR.InitPosY, interceptionPoint_R_O)

    # play steps for both players and for each step check who covered it first

    return stepsR


def IsNextSlotTheInitialPos(nextSlot, initialPos):
    return nextSlot[0] == initialPos[0] and nextSlot[1] == initialPos[1]



