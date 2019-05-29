from Entities import Slot, Strategy
from abc import abstractmethod

class CoverByQuarters_Strategy(Strategy):
    @abstractmethod
    def get_steps(self, agent_r, board_size = 50, agent_o = None):
        # This coverage strategy covers first the top left, then top right, then bottom left, then bottom right quarters of
        # the area.
        # While building this function, we assumed 100X100 dimensions

        next_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)
        # flag = (agent_r.InitPosY == boardSizeY - 1 and not (agent_r.InitPosX == 0))

        steps = []
        counter = 0

        while True:
            counter += 1

            if counter > 100000:
                print("Something is wrong, counter is too big!")
                print(steps)
                break

            if counter > 1 and (next_slot.row, next_slot.col) == (agent_r.InitPosX, agent_r.InitPosY):
                break

            steps.append((next_slot.row, next_slot.col))

            # TL Quarter
            if 0 <= next_slot.row < board_size / 2 and 0 <= next_slot.col < board_size / 2:
                if (next_slot.row, next_slot.col) == (board_size / 2 - 1, board_size / 2 - 1):
                    next_slot.row, next_slot.col = next_slot.go_east()
                    continue
                if next_slot.col == 0:
                    next_slot.row, next_slot.col = next_slot.go_north() if next_slot.row > 0 else next_slot.go_east()
                    continue
                if next_slot.row % 2 == 0 and next_slot.row < board_size / 2 - 2:  # An even line, not the last one
                    next_slot.row, next_slot.col = next_slot.go_east() if not next_slot.col == board_size / 2 - 1 else \
                        next_slot.go_south()
                    continue
                elif next_slot.row % 2 != 0 and not next_slot.row == board_size / 2 - 1:  # An odd line, not before last
                    next_slot.row, next_slot.col = next_slot.go_west() if not next_slot.col == 1 else next_slot.go_south()
                    continue
                elif next_slot.row % 2 == 0 and next_slot.row == board_size / 2 - 2:  # An even line, the last one
                    next_slot.row, next_slot.col = next_slot.go_south() if next_slot.col % 2 != 0 else next_slot.go_east()
                elif next_slot.row % 2 != 0 and next_slot.row == board_size / 2 - 1:  # An odd line, last line
                    next_slot.row, next_slot.col = next_slot.go_east() if next_slot.col % 2 != 0 else next_slot.go_north()
                else:
                    print("TL: Error occurred! Should not reach here!")
                continue

            # TR Quarter
            elif 0 <= next_slot.row < board_size / 2 and board_size / 2 <= next_slot.col < board_size:
                if (next_slot.row, next_slot.col) == (board_size / 2 - 1, board_size - 1):
                    next_slot.row, next_slot.col = next_slot.go_south()
                    continue
                elif next_slot.col % 2 == 0:
                    next_slot.row, next_slot.col = next_slot.go_north() if next_slot.row > 0 else next_slot.go_east()
                    continue
                elif next_slot.col % 2 != 0:
                    next_slot.row, next_slot.col = next_slot.go_south() if next_slot.row < board_size / 2 - 1 else \
                        next_slot.go_east()
                    continue
                else:
                    print("TR: Error occurred! Should not reach here!")
                continue

            # BL Quarter
            elif board_size / 2 <= next_slot.row < board_size and 0 <= next_slot.col < board_size / 2:
                if (next_slot.row, next_slot.col) == (board_size / 2, 0):  # last cell of quarter
                    next_slot.row, next_slot.col = next_slot.go_north()
                    continue
                elif next_slot.col % 2 == 0:  # an even column
                    next_slot.row, next_slot.col = next_slot.go_north() if next_slot.row > board_size / 2 else \
                        next_slot.go_west()
                    continue
                elif next_slot.col % 2 != 0:  # An odd line
                    next_slot.row, next_slot.col = next_slot.go_south() if next_slot.row < board_size - 1 else \
                        next_slot.go_west()
                    continue
                else:
                    print("BL: Error occurred! Should not reach here!")
                continue

            # BR Quarter
            else:
                if (next_slot.row, next_slot.col) == (board_size / 2, board_size / 2):  # last cell of quarter
                    next_slot.row, next_slot.col = next_slot.go_west()
                    continue
                elif next_slot.col == board_size - 1:
                    next_slot.row, next_slot.col = next_slot.go_south() if not next_slot.row == board_size - 1 else \
                        next_slot.go_west()
                    continue
                elif next_slot.row % 2 != 0 and not next_slot.row == board_size / 2 + 1:  # and odd line, not before last
                    next_slot.row, next_slot.col = next_slot.go_west() if next_slot.col > board_size / 2 else \
                        next_slot.go_north()
                    continue
                elif next_slot.row % 2 != 0 and next_slot.row == board_size / 2 + 1:  # and odd line, DO before last
                    next_slot.row, next_slot.col = next_slot.go_north() if next_slot.col % 2 == 0 else next_slot.go_west()
                    continue
                elif next_slot.row % 2 == 0 and not next_slot.row == board_size / 2:  # an even line, not last one
                    next_slot.row, next_slot.col = next_slot.go_east() if next_slot.col < board_size - 2 else \
                        next_slot.go_north()
                    continue
                elif next_slot.row % 2 == 0 and next_slot.row == board_size / 2:  # an even line, INDEED last one
                    next_slot.row, next_slot.col = next_slot.go_west() if next_slot.col % 2 == 0 else next_slot.go_south()
                    continue
                else:
                    print("BR: Error occurred! Should not reach here!")
                continue

        return steps
