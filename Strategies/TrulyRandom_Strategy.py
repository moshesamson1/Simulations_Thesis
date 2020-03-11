from Simulations_Thesis.Entities import Strategy, Slot, Agent
import random

def sublist(lst1, lst2):
    return all([(x in lst2) for x in lst1])
   # ls1 = [element for element in lst1 if element in lst2]
   # ls2 = [element for element in lst2 if element in lst1]
   # return ls1 == ls2

class TrulyRandomStrategy(Strategy):
    @classmethod
    def get_steps(self, agent_r: Agent, board_size=50, agent_o:Agent=None):
        covered_cells = []
        current_slot = Slot(agent_r.InitPosX, agent_r.InitPosY)
        covered_cells.append(current_slot)
        all_slots = [Slot(i,j) for i in range(board_size) for j in range(board_size)]
        while not sublist(all_slots, covered_cells):
            # randomly select move
            new_slot = current_slot
            while not (0 <= new_slot.row < board_size and 0 <= new_slot.col < board_size) or new_slot == current_slot:
                new_slot = current_slot.go(random.choice(['r','l','u','d']))

            # update accordingly
            covered_cells.append(new_slot)
            current_slot = new_slot
            print(len(set(covered_cells))/(board_size*board_size*1.0))
        print(covered_cells)
        print(len(covered_cells))
        exit(2)
        return covered_cells

