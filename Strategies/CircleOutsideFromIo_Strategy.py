from Simulations_Thesis.Entities import Slot, Strategy


class CircleOutsideFromIo_Strategy(Strategy):
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        """
        This function returns the coverage steps, choosing to start from the initial position of the opponent, io, and from
        there cover the world circling out.
        This function perform spiraling out simply: doesn't take steps in alternate sizes, and limiting the available slots
        to be in range
        :param self: the agent
        :param board_size:
        :param agent_o:
        :return:
        """

        fixed_io = Slot(agent_o.InitPosX, agent_o.InitPosY)
        self.steps.extend(
            Strategy.go_from_a_to_b(
                Slot(agent_r.InitPosX, agent_r.InitPosY),
                fixed_io))

        # cover the world, circling from this position out.
        current_slot = self.steps[-1]
        step_size = 1
        counter = 1

        b = [[0 for i in xrange(board_size)] for j in xrange(board_size)]
        b[current_slot.row][current_slot.col] = 1

        while sum(map(sum, b)) < board_size*board_size:
            # go horizontally right then vertically up
            for dir in ['r','u']:
                for _ in xrange(step_size):
                    current_slot = current_slot.go(dir)
                    counter += 1
                    if 0 <= current_slot.row < board_size and 0 <= current_slot.col < board_size:
                        self.steps.append(current_slot)
                        b[current_slot.row][current_slot.col] = 1

            step_size += 1
            # go horizontally left then vertically down
            for dir in ['l','d']:
                for _ in xrange(step_size):
                    current_slot = current_slot.go(dir)
                    counter += 1
                    if 0 <= current_slot.row < board_size and 0 <= current_slot.col < board_size:
                        self.steps.append(current_slot)
                        b[current_slot.row][current_slot.col] = 1
            step_size += 1
        return self.steps
