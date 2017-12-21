import numpy as np
from pyha import Hardware, simulate, sims_close
from pyhacores.filter import FIR
from scipy import signal

np.random.seed(0)  # reproduce tests


class ComplexFIR(Hardware):
    def __init__(self, taps):
        # registers
        self.fir = [FIR(taps), FIR(taps)]

        # constants (written in CAPS)
        self.DELAY = self.fir[0].DELAY
        self.TAPS = np.asarray(taps).tolist()

    def main(self, x):
        out = x
        out.real = self.fir[0].main(x.real)
        out.imag = self.fir[1].main(x.imag)
        return out

    def model_main(self, x):
        """ Golden output """
        return signal.lfilter(self.TAPS, [1.0], x)


def test_small():
    taps = signal.remez(8, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = ComplexFIR(taps)
    inp = np.random.uniform(-1, 1, 1024) + np.random.uniform(-1, 1, 1024) * 1j

    sims = simulate(dut, inp,
                    # simulations=['PYHA', 'RTL'], conversion_path='/home/gaspar/git/pyha_demo_project/conversion_src'
                    )

    # import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'], label='MODEL')
    # plt.plot(sims['PYHA'], label='PYHA')
    # plt.plot(sims['RTL'], label='RTL')
    # plt.legend()
    # plt.show()
    assert sims_close(sims)


def test_remez16():
    taps = signal.remez(16, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = ComplexFIR(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j

    sims = simulate(dut, inp)

    # import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'], label='MODEL')
    # plt.plot(sims['PYHA'], label='PYHA')
    # plt.plot(sims['RTL'], label='RTL')
    # plt.legend()
    # plt.show()
    assert sims_close(sims)
