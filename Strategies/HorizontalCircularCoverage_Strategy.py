from Entities import Strategy, Slot
from abc import abstractmethod

class HorizontalCircularCoverage_Strategy(Strategy):
    @abstractmethod
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        next_slot = (agent_r.InitPosX, agent_r.InitPosY)
        flag = (agent_r.InitPosX == board_size - 1)
        counter = 0

        # print "init pos {},{}: ".format(agent_r.InitPosX, agent_r.InitPosY)

        while True:
            counter += 1
            if counter > board_size * board_size:
                break

            self.steps.append(next_slot)
            # in the middle of moving from bottom rpw to top row
            if flag:
                if next_slot[1] == board_size - 1:
                    flag = False
                    next_slot = (next_slot[0] - 1, next_slot[1])
                else:
                    next_slot = (next_slot[0], next_slot[1] + 1)
                if next_slot == (agent_r.InitPosX, agent_r.InitPosY):
                    break
                continue
            # check if in last position, and start moving from last row to top row
            elif next_slot[0] == board_size - 1 - 1 and next_slot[1] == 0:
                flag = True
                next_slot = (next_slot[0] + 1, next_slot[1])

                if next_slot == (agent_r.InitPosX, agent_r.InitPosY):
                    break
                continue
            # update next slot
            elif next_slot[1] % 2 != 0:
                if next_slot[0] == 0:
                    next_slot = (next_slot[0], next_slot[1] - 1)
                else:
                    next_slot = (next_slot[0] - 1, next_slot[1])
            else:
                if next_slot[0] == board_size - 1 - 1:
                    next_slot = (next_slot[0], next_slot[1] - 1)
                else:
                    next_slot = (next_slot[0] + 1, next_slot[1])
                    if next_slot[0] > board_size - 1:
                        print "+1"

            if next_slot == Slot(agent_r.InitPosX, agent_r.InitPosY):
                break

        return self.steps
