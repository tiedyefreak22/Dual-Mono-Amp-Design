
import numpy as np
from scipy.optimize import root_scalar
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

def simulate_thd(triode, Vg_op, pk_pk, R_out, CCS_current, Vp_scan=None, f=1000, cycles=3, sampling_rate=100000):
    # Time vector
    t = np.linspace(0, cycles / f, int(cycles * sampling_rate))
    Vg_in = Vg_op + (pk_pk / 2) * np.sin(2 * np.pi * f * t)

    if Vp_scan is None:
        Vp_scan = np.linspace(min(triode.Vp_data), max(triode.Vp_data), 1000)

    interp = LinearNDInterpolator(np.column_stack((triode.Vp_data, triode.Vg_data)), triode.Ia_data)
    Vp_out = []

    for Vg in Vg_in:
        def residual(Vp):
            Ia = interp(Vp, Vg)
            if Ia is None or np.isnan(Ia):
                return 1e6
            return Ia - (CCS_current - Vp / R_out)

        found = False
        for i in range(len(Vp_scan) - 1):
            vp1, vp2 = Vp_scan[i], Vp_scan[i + 1]
            try:
                if np.sign(residual(vp1)) != np.sign(residual(vp2)):
                    sol = root_scalar(residual, bracket=[vp1, vp2], method='brentq')
                    if sol.converged:
                        Vp_out.append(sol.root)
                        found = True
                        break
            except:
                continue
        if not found:
            Vp_out.append(np.nan)

    Vp_out = np.array(Vp_out)
    valid = ~np.isnan(Vp_out)
    t_valid = t[valid]
    Vp_valid = Vp_out[valid]

    # FFT-based THD calculation
    yf = rfft(Vp_valid - np.mean(Vp_valid))
    freqs = rfftfreq(len(Vp_valid), 1 / sampling_rate)

    harmonics = np.abs(yf)
    fundamental = harmonics[1]  # skip DC
    harmonic_power = np.sum(harmonics[2:]**2)
    thd_ratio = np.sqrt(harmonic_power) / fundamental
    thd_db = 20 * np.log10(thd_ratio)

    # Plot waveform and spectrum
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].plot(t_valid * 1000, Vp_valid)
    axs[0].set_title("Simulated Plate Voltage vs Time")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Vp (V)")
    axs[0].grid(True)

    axs[1].stem(freqs[:10], 20 * np.log10(harmonics[:10]), use_line_collection=True)
    axs[1].set_title(f"Harmonics (THD: {thd_db:.2f} dB)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    return thd_ratio, thd_db, t_valid, Vp_valid
