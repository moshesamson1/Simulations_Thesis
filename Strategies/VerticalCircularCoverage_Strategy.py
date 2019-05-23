from Entities import Strategy, Slot
from abc import abstractmethod


class VerticalCircularCoverage_Strategy(Strategy):
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        next_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)

        flag = (agent_r.InitPosY == board_size - 1 and not (agent_r.InitPosX == 0))

        steps = []
        counter = 0

        while True:
            counter += 1
            if counter > 1000000000000:
                break

            steps.append(next_slot)
            # in the middle of moving from bottom row to top row
            if flag:
                next_slot = next_slot.go_north()
                if next_slot.row == 0:
                    flag = False
                if next_slot == Slot(agent_r.InitPosX, agent_r.InitPosY):
                    break
                continue
            # check if in last position, and start moving from last row to top row
            elif next_slot.row == board_size - 1 and next_slot.col == board_size - 1 - 1:
                flag = True
                next_slot = next_slot.go_east()
                if next_slot == Slot(agent_r.InitPosX, agent_r.InitPosY):
                    break
                continue
            # update next slot
            elif next_slot.row % 2 != 0:
                if next_slot.col == board_size - 1 - 1:
                    next_slot = next_slot.go_south()
                else:
                    next_slot = next_slot.go_east()
            else:
                if next_slot.col == 0:
                    next_slot = next_slot.go_south()
                else:
                    next_slot = next_slot.go_west()

            if next_slot == Slot(agent_r.InitPosX, agent_r.InitPosY):
                break

        return steps