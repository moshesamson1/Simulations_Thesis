from abc import abstractmethod

from Entities import Slot, Strategy, Agent


class GrowingDiagonalsCornerToCorner_Strategy(Strategy):
    @abstractmethod
    def get_steps(self, agent_r: Agent, board_size=50, agent_o: Agent = None):
        assert agent_o is not None

        # assert o's initial position is a corner
        o_init_slot = Slot(agent_o.InitPosX, agent_o.InitPosY)
        assert o_init_slot == (0, 0) or \
               o_init_slot == (0, board_size - 1) or \
               o_init_slot == (board_size - 1, 0) or \
               o_init_slot == (board_size - 1, board_size - 1)

        # go to the farthest corner
        steps_to_start = Strategy.go_from_a_to_b(a=Slot(agent_r.InitPosX, agent_r.InitPosY),
                                                 b=Strategy.get_adjacent_corner(
                                                     a=Slot(agent_o.InitPosX, agent_o.InitPosY),
                                                     board_size=board_size))  # or false!
        for i in steps_to_start:
            self.add_step(i)

        advance_counter = 1

        return self.steps
