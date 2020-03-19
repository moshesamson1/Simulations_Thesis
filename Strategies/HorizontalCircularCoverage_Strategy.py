from Simulations_Thesis.Entities import Strategy, Slot


class HorizontalCircularCoverage_Strategy(Strategy):
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        next_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)
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
                if next_slot.col == board_size - 1:
                    flag = False
                    next_slot = next_slot.go_north()
                else:
                    next_slot = next_slot.go_east()
                if next_slot == Slot(agent_r.InitPosX, agent_r.InitPosY):
                    break
                continue
            # check if in last position, and start moving from last row to top row
            elif next_slot.row == board_size - 1 - 1 and next_slot.col == 0:
                flag = True
                next_slot = next_slot.go_south()

                if next_slot == Slot(agent_r.InitPosX, agent_r.InitPosY):
                    break
                continue
            # update next slot
            elif next_slot.col % 2 != 0:
                if next_slot.row == 0:
                    next_slot = next_slot.go_west()
                else:
                    next_slot = next_slot.go_north()
            else:
                if next_slot.row == board_size - 1 - 1:
                    next_slot = next_slot.go_west()
                else:
                    next_slot = next_slot.go_south()
                    if next_slot.row > board_size - 1:
                        print("+1")

            if next_slot == Slot(agent_r.InitPosX, agent_r.InitPosY):
                break

        return self.steps
