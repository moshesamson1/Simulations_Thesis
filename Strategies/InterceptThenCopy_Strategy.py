from Simulations_Thesis.Entities import Strategy, Slot
from math import *
import numpy as np

def get_interception_point(steps_o, InitPosX, InitPosY):
    ip_x = -1
    ip_y = -1

    steps_counter = 0
    for step in steps_o:
        steps_counter += 1
        distance_from_i_r = fabs(step.row - InitPosX) + fabs(step.col - InitPosY) + 1
        if fabs(steps_counter - distance_from_i_r) <= 1:
            ip_x, ip_y = step.row, step.col
            break

    assert (ip_x != -1 and ip_y != -1)

    return Slot(ip_x, ip_y), fabs(ip_x - InitPosX) + fabs(ip_y - InitPosY)


class InterceptThenCopy_Strategy(Strategy):
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        steps_o = Strategy.get_strategy_from_enum(agent_o.StrategyEnum).get_steps(agent_o, board_size, board_size)

        # Find interception point
        (interceptionPoint_R_O, distance) = get_interception_point(steps_o, agent_r.InitPosX, agent_r.InitPosY)

        steps_r = run_agent_over_board_interception_strategy(steps_o, agent_r.InitPosX, agent_r.InitPosY,
                                                             interceptionPoint_R_O)

        # play steps for both players and for each step check who covered it first

        return steps_r


def run_agent_over_board_interception_strategy(steps_o, i_r_x, i_r_y, interception_point):
    distance2_interception_point = fabs(i_r_x - interception_point.row) + fabs(i_r_y - interception_point.col)
    next_slot = Slot(i_r_x, i_r_y)
    steps = [next_slot]
    counter = 0

    if np.sign(interception_point.row - i_r_x) >= 0:
        x_sign = 1
    else:
        x_sign = -1

    if np.sign(interception_point.col - i_r_y) >= 0:
        y_sign = 1
    else:
        y_sign = -1

    while True:
        if counter >= len(steps_o) - 1:
            break

        # first, go to interception point
        if counter < distance2_interception_point:
            for x_step in range(i_r_x + x_sign, interception_point.row + x_sign, x_sign):
                counter += 1
                next_slot = Slot(x_step, next_slot.col)
                steps.append(next_slot)
            for y_step in range(i_r_y + y_sign, interception_point.col + y_sign, y_sign):
                counter += 1
                next_slot = Slot(next_slot.row, y_step)
                steps.append(next_slot)
        # then, play as O plays
        else:
            counter += 1
            next_slot = steps_o[counter]
            steps.append(next_slot)

    return steps



