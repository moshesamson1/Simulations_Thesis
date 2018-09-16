from Entities import Slot, Strategy
from abc import abstractmethod


class VerticalCoverageFromCornerFarthestFromIo_Strategy(Strategy):
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        """
            This function returns the coverage self.steps,when covering knowing io, and start covering vertically from the farthest
            corner.
            :param self:
            :param board_size:
            :param agent_o:
            :return: the coverage self.steps.
            """
        assert agent_o is not None

        # go to the farthest corner
        self.steps.extend(Strategy.go_from_a_to_b(a=Slot(agent_r.InitPosX, agent_r.InitPosY),
                                                  b=Strategy.get_farthest_corner(
                                                      a=Slot(agent_o.InitPosX, agent_o.InitPosY),
                                                      board_size=board_size)))

        # from there, cover vertically
        current_slot = self.steps[-1]
        v_dir = 'u' if current_slot.row == board_size - 1 else 'd'
        h_dir = 'l' if current_slot.col == board_size - 1 else 'r'
        counter = 1
        while counter <= board_size * board_size:
            if v_dir == 'u':
                while current_slot.row > 0:
                    current_slot = current_slot.go_north()
                    self.steps.append(current_slot)
                    counter += 1

                if counter == board_size * board_size:
                    break

                current_slot = current_slot.go_west() if h_dir == 'l' else current_slot.go_east()
                v_dir = 'd'
                self.steps.append(current_slot)
                counter += 1
                continue
            else:
                while current_slot.row < board_size - 1:
                    current_slot = current_slot.go_south()
                    self.steps.append(current_slot)
                    counter += 1

                if counter == board_size * board_size:
                    break

                current_slot = current_slot.go_west() if h_dir == 'l' else current_slot.go_east()
                v_dir = 'u'
                self.steps.append(current_slot)
                counter += 1
                continue

        return self.steps
