import numpy as np
from pyha import Hardware, simulate, sims_close, ComplexSfix, Sfix, hardware_sims_equal
from pyhacores.filter import FIR
from scipy import signal
from copy import copy

np.random.seed(0)  # reproduce tests


class ComplexFIR(Hardware):
    def __init__(self, taps):
        # registers
        self.fir = [FIR(taps), FIR(taps)]
        # self.outreg = ComplexSfix()

        # constants (written in CAPS)
        self.DELAY = self.fir[0].DELAY
        self.TAPS = np.asarray(taps).tolist()

    def main(self, x):
        # real = self.fir[0].main(x.real)
        # imag = self.fir[1].main(x.imag)
        # ret = ComplexSfix(real, imag)
        # return ret

        out = x
        out.real = self.fir[0].main(x.real)
        out.imag = self.fir[1].main(x.imag)
        return out

        # self.outreg.real = self.fir[0].main(x.real)
        # self.outreg.imag = self.fir[1].main(x.imag)
        # return self.outreg

    def model_main(self, x_full):
        """ Golden output """
        from scipy.signal import lfilter
        return lfilter(self.TAPS, [1.0], x_full)


# class ComplexFIR(Hardware):
#     def __init__(self, taps):
#         # registers
#         self.fir = [FIR(taps), FIR(taps)]
#         self.outreg = ComplexSfix()
#
#         # constants (written in CAPS)
#         self.DELAY = self.fir[0].DELAY
#         self.TAPS = np.asarray(taps).tolist()
#
#     def main(self, x):
#         # out = x
#         # out.real = self.fir[0].main(x.real)
#         # out.imag = self.fir[1].main(x.imag)
#         # return out
#         self.outreg.real = self.fir[0].main(x.real)
#         self.outreg.imag = self.fir[1].main(x.imag)
#         return self.outreg
#
#     def model_main(self, x):
#         """ Golden output """
#         return signal.lfilter(self.TAPS, [1.0], x)


# def test_debug():
#     np.random.seed(0)  # reproduce tests
#     inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j
#     # inp *= 0.75
#
#     taps = signal.remez(128, [0, 0.2, 0.25, 0.5], [1, 0])
#     dut = ComplexFIR(taps)
#
#     sims = simulate(dut, inp, simulations=['PYHA'])  # run all simulations



# def test_bugg():
#     # get imulse response of the filter
#     inp = [0.0 + 0.0j] * 64
#     inp[0] = 1.0 + 1.0j
#     taps = signal.remez(128, [0, 0.2, 0.25, 0.5], [1, 0])
#
#     dut = ComplexFIR(taps)
#     sims = simulate(dut,  # pyha model
#                     inp,  # input to the 'main' function
#                     simulations=['MODEL', 'PYHA', 'RTL'],
#                     conversion_path='/home/gaspar/git/pyha_demo_project/conversion_src'
#                     )
#
#     assert hardware_sims_equal(sims)


def test_demo():
    taps = signal.remez(8, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = ComplexFIR(taps)
    input = [0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j]

    sims = simulate(dut, input,
                    simulations=['MODEL', 'PYHA', 'RTL'],
                    conversion_path='/home/gaspar/git/pyha_demo_project/conversion_src'
                    )

    assert sims_close(sims)


def test_remez16_random():
    taps = signal.remez(16, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = ComplexFIR(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])
    assert sims_close(sims)
