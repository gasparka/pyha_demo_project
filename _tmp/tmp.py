import numpy as np
from pyha import Hardware, simulate, sims_close, Sfix


class Lookup(Hardware):
    def __init__(self):
        self.mem = np.random.uniform(-1, 1, 5)
        self.counter = Sfix(0, 4, 0)

    def main(self, x):
        ret = self.mem[int(self.counter)]


        counter_next = self.counter + 1
        if counter_next >= len(self.mem):
            counter_next = 0

        self.counter = counter_next
        return ret


def test_remez64_random_notebook():
    dut = Lookup()

    inp = [0] * 16
    sims = simulate(dut, inp, simulations=[
        'MODEL_PYHA',
                                           'PYHA', 'RTL'])
    assert sims_close(sims, rtol=1e-3)



# class Lookup(Hardware):
#     def __init__(self):
#         self.mem = np.random.uniform(-1, 1, 5).tolist()
#         self.counter = 0
#
#     def main(self, x):
#         ret = self.mem[self.counter]
#
#         next_counter = self.counter + 1
#         if next_counter >= len(self.mem):
#             next_counter = 0
#
#         self.counter = next_counter
#
#         return ret