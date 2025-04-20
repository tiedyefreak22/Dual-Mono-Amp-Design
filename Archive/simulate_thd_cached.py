
import numpy as np
from scipy.optimize import root_scalar
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

def simulate_thd(triode, Vg_op, pk_pk, R_out, CCS_current, Vp_scan=None, f=1000, cycles=3, sampling_rate=100000):
    t = np.linspace(0, cycles / f, int(cycles * sampling_rate))
    Vg_in = Vg_op + (pk_pk / 2) * np.sin(2 * np.pi * f * t)

    if Vp_scan is None:
        Vp_scan = np.linspace(min(triode.Vp_data), max(triode.Vp_data), 1000)

    interp = LinearNDInterpolator(np.column_stack((triode.Vp_data, triode.Vg_data)), triode.Ia_data)
    Vp_out = []

    last_vp = 150
    fallback_used = False

    for Vg in Vg_in:
        def residual(Vp):
            Ia = interp(Vp, Vg)
            if Ia is None or np.isnan(Ia):
                return 1e6
            return Ia - (CCS_current - Vp / R_out)

        found = False
        try:
            sol = root_scalar(
                residual,
                bracket=[last_vp - 10, last_vp + 10],
                method='brentq'
            )
            if sol.converged:
                last_vp = sol.root
                Vp_out.append(sol.root)
                found = True
        except:
            pass

        if not found:
            # fallback bracket
            try:
                sol = root_scalar(residual, bracket=[0, 400], method='brentq')
                if sol.converged:
                    last_vp = sol.root
                    Vp_out.append(sol.root)
                    fallback_used = True
                    continue
            except:
                Vp_out.append(np.nan)

    Vp_out = np.array(Vp_out)
    valid = ~np.isnan(Vp_out)

    if fallback_used:
        print(f"⚠️ Fallback root bracket used for pk_pk={pk_pk}")

    if np.sum(valid) < 10:
        print(f"❌ Insufficient valid data points for pk_pk={pk_pk}")
        return np.nan, np.nan, t[valid], Vp_out[valid]

    t_valid = t[valid]
    Vp_valid = Vp_out[valid]

    yf = rfft(Vp_valid - np.mean(Vp_valid))
    freqs = rfftfreq(len(Vp_valid), 1 / sampling_rate)

    harmonics = np.abs(yf)
    fundamental = harmonics[1] if len(harmonics) > 1 else 1e-6
    harmonic_power = np.sum(harmonics[2:]**2)
    thd_ratio = np.sqrt(harmonic_power) / fundamental
    thd_db = 20 * np.log10(thd_ratio)

    # Plot only if enough points
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(t_valid * 1000, Vp_valid)
    axs[0].set_title("Simulated Plate Voltage vs Time")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Vp (V)")
    axs[0].grid(True)

    axs[1].stem(freqs[:10], 20 * np.log10(harmonics[:10]))
    axs[1].set_title(f"Harmonics (THD: {thd_db:.2f} dB)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

    return thd_ratio, thd_db, t_valid, Vp_valid
