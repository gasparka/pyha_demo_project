import numpy as np
import pytest
import scipy.signal
from pyhacores.filter import FIR
from pyha import Hardware, simulate, sims_close, Complex


class BasebandFilter(Hardware):
    def __init__(self, taps):
        self.fir = [FIR(taps), FIR(taps)]
        self.TAPS = taps
        self.DELAY = self.fir[0].DELAY

    def main(self, complex_in):
        """ Apply FIR filters to 'real' and 'imag' channels """
        real = self.fir[0].main(complex_in.real)
        imag = self.fir[1].main(complex_in.imag)
        return Complex(real, imag)

    def model_main(self, complex_in_list):
        """ Golden output """
        return scipy.signal.lfilter(self.TAPS, [1.0], complex_in_list)


def test_remez64_random_notebook():
    np.random.seed(0)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j
    # inp *= 0.75

    taps = scipy.signal.remez(64, [0, 0.2, 0.275, 0.5], [1, 0])
    dut = BasebandFilter(taps)

    sims = simulate(dut, inp)
    assert sims_close(sims, rtol=1e-3)


def test_remez64_random_too_much_gain():
    pytest.xfail('Gain is too large, filter saturates')
    np.random.seed(0)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j
    # inp *= 0.75

    taps = scipy.signal.remez(64, [0, 0.2, 0.275, 0.5], [1, 0])
    dut = BasebandFilter(taps)

    sims = simulate(dut, inp)
    assert sims_close(sims, rtol=1e-3)


def test_remez16_random():
    np.random.seed(0)
    taps = scipy.signal.remez(16, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = BasebandFilter(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j

    sims = simulate(dut, inp)
    assert sims_close(sims)


def test_remez32_random():
    np.random.seed(0)
    taps = scipy.signal.remez(32, [0, 0.4, 0.45, 0.5], [1, 0])
    dut = BasebandFilter(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j
    inp *= 0.75

    sims = simulate(dut, inp)
    assert sims_close(sims, rtol=1e-3)


def test_remez64_random():
    np.random.seed(0)
    taps = scipy.signal.remez(64, [0, 0.05, 0.15, 0.5], [1, 0])
    dut = BasebandFilter(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j

    sims = simulate(dut, inp)
    assert sims_close(sims, rtol=1e-2)
