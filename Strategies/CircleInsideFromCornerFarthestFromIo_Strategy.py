from Entities import Slot, Strategy
from abc import  abstractmethod

class CircleInsideFromCornerFarthestFromIo_Strategy(Strategy):
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        """
            This function return the agent steps, when deciding to cover the world, spiraling from outside to inside.
            Note: this coverage method is not optimal!
            :param agent: the agent covering the world
            :param board_size: self explanatory
            :return: list of steps
            """

        next_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)

        # start by going toward the closest corner
        if next_slot.row < board_size / 2:
            while next_slot.row > 0:
                self.steps.append(next_slot)
                next_slot = next_slot.go_north()
                continue
        else:
            while next_slot.row < board_size - 1:
                self.steps.append(next_slot)
                next_slot = next_slot.go_south()
                continue

        if next_slot.col < board_size / 2:
            while next_slot.col > 0:
                self.steps.append(next_slot)
                next_slot = next_slot.go_west()
                continue
        else:
            while next_slot.col < board_size - 1:
                self.steps.append(next_slot)
                next_slot = next_slot.go_east()
                continue
        self.steps.append(next_slot)

        # after reaching the closest-to-start corner, start covering the world, counter clockwise
        shallow_slots = [[0 for x in range(board_size)] for y in range(board_size)]
        shallow_slots[next_slot.row][next_slot.col] = 1
        dist_from_edge = 0

        while dist_from_edge < board_size / 2:
            if next_slot.row + dist_from_edge < board_size - 1 and next_slot.col == dist_from_edge:
                direction = 's'
            elif next_slot.row + dist_from_edge == board_size - 1 and next_slot.col + dist_from_edge < board_size - 1:
                direction = 'e'
            elif next_slot.row > dist_from_edge and next_slot.col + dist_from_edge == board_size - 1:
                direction = 'n'
            elif next_slot.row == dist_from_edge and next_slot.col + dist_from_edge >= board_size - 1:
                direction = 'w'

            if direction == 's':
                new_slot = next_slot.go_south()
            elif direction == 'e':
                new_slot = next_slot.go_east()
            elif direction == 'n':
                new_slot = next_slot.go_north()
            elif direction == 'w':
                new_slot = next_slot.go_west()

            if shallow_slots[new_slot.row][new_slot.col] == 1:
                dist_from_edge += 1
                continue
            else:
                next_slot = new_slot

            self.steps.append(next_slot)
            shallow_slots[next_slot.row][next_slot.col] = 1

        return self.steps