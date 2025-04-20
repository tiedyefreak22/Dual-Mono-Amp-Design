
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import LinearNDInterpolator

def simulate_thd_from_grid(triode, Vg_op, pk_pk, R_out, CCS_current, f=1000, cycles=1.5, sampling_rate=10000):
    t = np.linspace(0, cycles / f, int(cycles * sampling_rate))
    Vg_in = Vg_op + (pk_pk / 2) * np.sin(2 * np.pi * f * t)

    points = np.column_stack((triode.Vp_grid.ravel(), triode.Vg_grid.ravel()))
    values = triode.Ia_grid.ravel()
    interp = LinearNDInterpolator(points, values)

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
            sol = root_scalar(residual, bracket=[last_vp - 10, last_vp + 10], method='brentq')
            if sol.converged:
                last_vp = sol.root
                Vp_out.append(sol.root)
                found = True
        except:
            pass

        if not found:
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
    if np.sum(valid) < 10:
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

    return thd_ratio, thd_db, t_valid, Vp_valid


def optimize_operating_point(triode, pk_pk=2.0, R_out=1e6,
                              Vg_range=(-3.0, 0.0, 10),
                              CCS_range=(0.5, 3.0, 10),
                              gain_threshold=0.1,
                              sampling_rate=10000,
                              f=1000,
                              cycles=1.5):
    Vg_vals = np.linspace(*Vg_range)
    CCS_vals = np.linspace(*CCS_range)
    
    THD_map = np.full((len(Vg_vals), len(CCS_vals)), np.nan)
    Gain_map = np.full_like(THD_map, np.nan)

    best_thd = np.inf
    best_params = None

    for i, Vg_op in enumerate(Vg_vals):
        for j, CCS_current in enumerate(CCS_vals):
            try:
                thd, thd_db, t, vp = simulate_thd_from_grid(
                    triode, Vg_op, pk_pk, R_out, CCS_current,
                    sampling_rate=sampling_rate, f=f, cycles=cycles
                )
                if np.isnan(thd) or len(vp) < 2:
                    continue
                gain = (np.nanmax(vp) - np.nanmin(vp)) / pk_pk

                THD_map[i, j] = thd
                Gain_map[i, j] = gain

                if gain >= gain_threshold and thd < best_thd:
                    best_thd = thd
                    best_params = (Vg_op, CCS_current, gain, thd)

            except Exception as e:
                print(f"Error at Vg={Vg_op:.2f}, CCS={CCS_current:.2f}: {e}")
                continue

    # Plot THD heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(THD_map * 100, extent=[CCS_vals[0], CCS_vals[-1], Vg_vals[0], Vg_vals[-1]],
                   origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label="THD (%)")
    ax.set_xlabel("CCS Current (mA)")
    ax.set_ylabel("Grid Bias Vg_op (V)")
    ax.set_title(f"THD Heatmap (pk-pk={pk_pk}V)")

    if best_params:
        ax.plot(best_params[1], best_params[0], 'ro', label="Best")
        ax.legend()
        print(f"✅ Best Operating Point: Vg = {best_params[0]:.2f} V, CCS = {best_params[1]:.2f} mA")
        print(f"   Gain = {best_params[2]:.2f}, THD = {best_params[3]*100:.2f}%")
    else:
        print("⚠️ No valid solution found.")

    plt.tight_layout()
    plt.show()
    
    return best_params, Vg_vals, CCS_vals, THD_map, Gain_map
