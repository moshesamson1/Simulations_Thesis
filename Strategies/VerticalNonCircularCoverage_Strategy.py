from Simulations_Thesis.Entities import Slot, Strategy
from abc import abstractmethod

class VerticalNonCircularCoverage_Strategy(Strategy):
    @abstractmethod
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        """
            Returns a non-circular vertical coverage, starting from agent_r's initial position to top-right position, then cover
            all cells from top-right to bottom-left, without repeating cells (there are some assumption regarding the initial
            position
            :param agent_r: the given agent_r
            :param board_size: board's width
            :param board_size: board's height
            :return: a set of steps covering the whole board
            """
        steps = []
        next_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)
        turning_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)
        reaching_to_farthest_corner = True
        up_or_down = 'd'

        # assert call
        assert agent_r.InitPosX % 2 == 0
        while True:
            steps.append(next_slot)
            if len(steps) >= board_size * board_size:
                break

            # Check if we agent_r reached the farthest corner of the board
            if next_slot == Slot(board_size - 1, board_size - 1):
                reaching_to_farthest_corner = False

            if reaching_to_farthest_corner:
                if next_slot.row < board_size - 1:
                    next_slot = next_slot.go_south()
                    continue
                elif next_slot.col < board_size - 1:
                    next_slot = next_slot.go_east()
                    continue
            else:
                if next_slot.col > agent_r.InitPosY:
                    if next_slot.col % 2 != 0:
                        next_slot = next_slot.go_north() if next_slot.row > 0 else next_slot.go_west()
                    else:
                        next_slot = next_slot.go_south() if next_slot.row < board_size - 2 else next_slot.go_west()
                    continue
                else:
                    if up_or_down == 'd':
                        if next_slot.go_south() != turning_slot:
                            if next_slot.row < board_size - 1:
                                next_slot = next_slot.go_south()
                            else:
                                next_slot = next_slot.go_west()
                                up_or_down = 'u'
                            continue
                        else:
                            turning_slot = turning_slot.go_west().go_west().go_north().go_north()
                            steps.append(next_slot.go_west())
                            steps.append(next_slot.go_west().go_south())
                            next_slot = next_slot.go_west().go_south().go_south()
                            continue
                    else:
                        if next_slot != turning_slot:
                            if next_slot.row > 0:
                                next_slot = next_slot.go_north()
                            else:
                                next_slot = next_slot.go_west()
                                up_or_down = 'd'
                            continue
                        else:
                            steps.append(next_slot.go_east())
                            steps.append(next_slot.go_east().go_north())
                            next_slot = next_slot.go_east().go_north().go_north()
                            continue

        return steps[:board_size * board_size]