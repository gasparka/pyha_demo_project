import numpy as np
from pyha import Hardware, simulate, sims_close, hardware_sims_equal, ComplexSfix
from pyhacores.filter import FIR
from scipy import signal

np.random.seed(0)  # reproduce tests


class ComplexFIR(Hardware):
    def __init__(self, taps):
        # registers
        self.fir = [FIR(taps), FIR(taps)]
        # self.outreg = ComplexSfix(right=-7)

        # constants (written in CAPS)
        self.DELAY = self.fir[0].DELAY
        self.TAPS = np.asarray(taps).tolist()

    def main(self, x):
        out = x
        out.real = self.fir[0].main(x.real)
        out.imag = self.fir[1].main(x.imag)
        return out
        # self.outreg = out
        # return self.outreg

    def model_main(self, x):
        """ Golden output """
        return signal.lfilter(self.TAPS, [1.0], x)


def test_small():
    taps = signal.remez(8, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = ComplexFIR(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL', 'GATE'],
                    conversion_path='/home/gaspar/git/pyha_demo_project/conversion_src')

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

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'],
                    conversion_path='/home/gaspar/git/pyha/playground')

    # import matplotlib.pyplot as plt
    # plt.plot(sims['MODEL'], label='MODEL')
    # plt.plot(sims['PYHA'], label='PYHA')
    # plt.plot(sims['RTL'], label='RTL')
    # plt.legend()
    # plt.show()
    assert sims_close(sims)


def test_junk():
    taps = signal.remez(16, [0, 0.1, 0.2, 0.5], [1, 0])
    # get imulse response of the filter
    inp = [0.0 + 0.0j] * 512
    inp[0] = 1.0 + 1.0j

    dut = ComplexFIR(taps)
    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])  # run all simulations
    assert hardware_sims_equal(sims)
