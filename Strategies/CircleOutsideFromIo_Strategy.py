from Entities import Slot, Strategy


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

        # Make assertions over opp initial position: we want it to be with even column and odd row.
        # Then, go to the fixed initial position
        fixed_io = Slot(agent_o.InitPosX, agent_o.InitPosY)
        if fixed_io.row % 2 == 0:
            fixed_io.row += 1
        if fixed_io.col % 2 != 0:
            fixed_io.col -= 1
        self.steps.extend(
            Strategy.go_from_a_to_b(
                Slot(agent_r.InitPosX, agent_r.InitPosY),
                fixed_io))

        # cover the world, circling from this position out. Taking big steps to one direction to compensate for the other.
        current_slot = self.steps[-1]
        step_size = 1
        counter = 1

        while counter < board_size * board_size:
            # go horizontally
            for _ in xrange(step_size):
                current_slot = current_slot.go('r')
                counter += 1
                if current_slot.row < board_size and current_slot.col < board_size:
                    self.steps.append(current_slot)
            # go vertically
            for _ in xrange(step_size):
                current_slot = current_slot.go('u')
                counter += 1
                if current_slot.row < board_size and current_slot.col < board_size:
                    self.steps.append(current_slot)

            step_size += 1
            # go horizontally
            for _ in xrange(step_size):
                current_slot = current_slot.go('l')
                counter += 1
                if current_slot.row < board_size and current_slot.col < board_size:
                    self.steps.append(current_slot)
            # go vertically
            for _ in xrange(step_size):
                current_slot = current_slot.go('d')
                counter += 1
                if current_slot.row < board_size and current_slot.col < board_size:
                    self.steps.append(current_slot)

        return self.steps
