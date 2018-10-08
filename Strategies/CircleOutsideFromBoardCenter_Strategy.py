from Entities import Strategy, Slot
from abc import abstractmethod

class CircleOutsideFromBoardCenter_Strategy(Strategy):
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        """
            This function return the agent_r steps, when deciding to cover the world, spiraling from inside to outside.
            Note: this coverage method is not optimal!
            :param agent_r: the agent covering the world
            :param board_size: self explanatory
            :return: list of steps
            """

        next_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)
        # next_slot = Slot(36,6)

        # start by going toward the center
        if next_slot.row < board_size / 2:
            while next_slot.row < board_size / 2:
                next_slot = next_slot.go_south()
                self.steps.append(next_slot)
                continue
        else:
            while next_slot.row > board_size / 2:
                next_slot = next_slot.go_north()
                self.steps.append(next_slot)
                continue

        if next_slot.col < board_size / 2:
            while next_slot.col < board_size / 2:
                next_slot = next_slot.go_east()
                self.steps.append(next_slot)
                continue
        else:
            while next_slot.col > board_size / 2:
                next_slot = next_slot.go_west()
                self.steps.append(next_slot)
                continue

        # after reaching the center, start covering, counter clockwise
        circ = 0
        counter = 1
        while circ < board_size:
            circ += 1
            if circ < board_size:
                for _ in range(circ):
                    next_slot = next_slot.go_west()
                    self.steps.append(next_slot)
                    counter += 1
            if circ < board_size:
                for _ in range(circ):
                    next_slot = next_slot.go_north()
                    self.steps.append(next_slot)
                    counter += 1

            circ += 1
            if circ < board_size:
                for _ in range(circ):
                    next_slot = next_slot.go_east()
                    self.steps.append(next_slot)
                    counter += 1
            if circ < board_size:
                for _ in range(circ):
                    next_slot = next_slot.go_south()
                    self.steps.append(next_slot)
                    counter += 1

        for step in self.steps:
            if step.row > board_size-1 or step.col > board_size-1 or step.row < 0 or step.col < 0:
                print "Error while circling outside from board center: reached unavailable position"
                break
        return self.steps
