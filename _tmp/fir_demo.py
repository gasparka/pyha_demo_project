from scipy import signal
from pyha import Hardware, Sfix, simulate, sims_close
import numpy as np
from copy import deepcopy


class FIR(Hardware):
    def __init__(self, taps):
        self.DELAY = 2
        self.TAPS = np.array(taps).tolist()

        # registers
        self.acc = [Sfix(left=1, right=-23)] * len(taps)
        self.out = Sfix(left=0, right=-17, overflow_style='saturate')

    def main(self, x):
        """ Transposed FIR structure """

        old_acc = deepcopy(self.acc)
        self.acc[0] = x * self.TAPS[-1]
        for i in range(1, len(self.acc)):
            self.acc[i] = old_acc[i - 1] + x * self.TAPS[len(self.TAPS) - 1 - i]

        self.out = self.acc[-1]
        return self.out

    def model_main(self, x):
        return signal.lfilter(self.TAPS, [1.0], x)


def test_simple():
    taps = [0.01, 0.02]
    dut = FIR(taps)
    inp = [0.1, 0.2, 0.3, 0.4]

    sims = simulate(dut, inp, simulations=['MODEL', 'MODEL_PYHA', 'PYHA', 'RTL', 'GATE'])

    assert sims_close(sims)


    # def main(self, x):
    #     """ Transposed FIR structure """
    #
    #     acc_old = deepcopy(self.acc)
    #     self.acc[0] = x * self.TAPS[-1]
    #     for i in range(1, len(self.acc)):
    #         self.acc[i] = acc_old[i - 1] + x * self.TAPS[len(self.TAPS) - 1 - i]
    #
    #     self.out = self.acc[-1]
    #     return self.out