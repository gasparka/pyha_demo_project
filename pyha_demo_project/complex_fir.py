import numpy as np
import scipy.signal
from pyhacores.filter import FIR
from pyha import Hardware, simulate, sims_close, Complex

np.random.seed(0)  # reproduce tests


def plot_freqz(b):
    import matplotlib.pyplot as plt
    from scipy import signal
    w, h = signal.freqz(b)

    fig, ax1 = plt.subplots(1, 1)
    plt.title('Digital filter frequency response')
    ax1.plot(w / np.pi, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency')
    plt.grid()
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w / np.pi, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.axis('tight')
    plt.show()


class BasebandFilter(Hardware):
    def __init__(self, taps):
        self.fir = [FIR(taps), FIR(taps)]

        # constants (written in CAPS)
        self.TAPS = taps
        self.DELAY = self.fir[0].DELAY

    def main(self, x):
        """ Apply FIR filter to 'real' and 'imag' channels """
        real = self.fir[0].main(x.real)
        imag = self.fir[1].main(x.imag)
        return Complex(real, imag)

    def model_main(self, x):
        """ Golden output """
        return scipy.signal.lfilter(self.TAPS, [1.0], x)


def test_demo():
    taps = scipy.signal.remez(8, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = BasebandFilter(taps)
    input = [0.1 + 0.1j, 0.2 + 0.2j, 0.3 + 0.3j]

    sims = simulate(dut, input,
                    simulations=['MODEL', 'PYHA', 'RTL'],
                    conversion_path='/home/gaspar/git/pyha_demo_project/conversion_src'
                    )

    assert sims_close(sims)


def test_remez16_random():
    taps = scipy.signal.remez(16, [0, 0.1, 0.2, 0.5], [1, 0])
    dut = BasebandFilter(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])
    assert sims_close(sims)


def test_remez32_random():
    taps = scipy.signal.remez(32, [0, 0.4, 0.45, 0.5], [1, 0])
    dut = BasebandFilter(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j
    inp *= 0.75

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])
    assert sims_close(sims)


def test_remez64_random():
    taps = scipy.signal.remez(64, [0, 0.05, 0.1, 0.5], [1, 0])
    dut = BasebandFilter(taps)
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL'])
    assert sims_close(sims, rtol=1e-2)

def test_remez128_random_notebook():
    inp = np.random.uniform(-1, 1, 512) + np.random.uniform(-1, 1, 512) * 1j
    inp *= 0.75

    taps = scipy.signal.remez(128, [0, 0.2, 0.25, 0.5], [1, 0])
    dut = BasebandFilter(taps)

    sims = simulate(dut, inp, simulations=['MODEL', 'PYHA', 'RTL', 'GATE'],
                    conversion_path='/home/gaspar/git/pyha_demo_project/conversion_src'
                    )  # run all simulations
    assert sims_close(sims, rtol=1e-3)
