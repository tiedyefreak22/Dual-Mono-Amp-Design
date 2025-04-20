
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import root_scalar
from scipy.fft import rfft, rfftfreq

def simulate_dc_coupled_chain(
    triode1, triode2,
    Vg1_op, CCS1_current,
    R_out1, R_out2,
    pk_pk1,
    f=1000, cycles=1.5, sampling_rate=10000
):
    t = np.linspace(0, cycles / f, int(cycles * sampling_rate))
    Vg1_in = Vg1_op + (pk_pk1 / 2) * np.sin(2 * np.pi * f * t)

    # Interpolators for both stages from their 3D manifolds
    interp1 = LinearNDInterpolator(
        np.column_stack((triode1.Vp_grid.ravel(), triode1.Vg_grid.ravel())),
        triode1.Ia_grid.ravel()
    )
    interp2 = LinearNDInterpolator(
        np.column_stack((triode2.Vp_grid.ravel(), triode2.Vg_grid.ravel())),
        triode2.Ia_grid.ravel()
    )

    # First stage: Vp1 as function of Vg1
    Vp1_out = []
    last_vp1 = 150
    for Vg1 in Vg1_in:
        def residual(Vp1):
            Ia = interp1(Vp1, Vg1)
            if Ia is None or np.isnan(Ia):
                return 1e6
            return Ia - (CCS1_current - Vp1 / R_out1)
        try:
            sol = root_scalar(residual, bracket=[last_vp1 - 10, last_vp1 + 10], method='brentq')
            last_vp1 = sol.root
            Vp1_out.append(sol.root)
        except:
            try:
                sol = root_scalar(residual, bracket=[0, 400], method='brentq')
                last_vp1 = sol.root if sol.converged else last_vp1
                Vp1_out.append(sol.root if sol.converged else np.nan)
            except:
                Vp1_out.append(np.nan)

    Vp1_out = np.array(Vp1_out)
    Vg2_in = Vp1_out  # DC-coupled
    CCS2_current = np.nanmedian((Vp1_out / R_out1))  # estimate for now

    Vp2_out = []
    last_vp2 = 150
    for Vg2 in Vg2_in:
        def residual2(Vp2):
            Ia = interp2(Vp2, Vg2)
            if Ia is None or np.isnan(Ia):
                return 1e6
            return Ia - (CCS2_current - Vp2 / R_out2)
        try:
            sol = root_scalar(residual2, bracket=[last_vp2 - 10, last_vp2 + 10], method='brentq')
            last_vp2 = sol.root
            Vp2_out.append(sol.root)
        except:
            try:
                sol = root_scalar(residual2, bracket=[0, 400], method='brentq')
                last_vp2 = sol.root if sol.converged else last_vp2
                Vp2_out.append(sol.root if sol.converged else np.nan)
            except:
                Vp2_out.append(np.nan)

    Vp2_out = np.array(Vp2_out)
    valid = ~np.isnan(Vp2_out)
    if np.sum(valid) < 10:
        return np.nan, np.nan, t[valid], Vp2_out[valid]

    t_valid = t[valid]
    Vp_valid = Vp2_out[valid]
    yf = rfft(Vp_valid - np.mean(Vp_valid))
    freqs = rfftfreq(len(Vp_valid), 1 / sampling_rate)
    harmonics = np.abs(yf)
    fundamental = harmonics[1] if len(harmonics) > 1 else 1e-6
    harmonic_power = np.sum(harmonics[2:]**2)
    thd_ratio = np.sqrt(harmonic_power) / fundamental
    thd_db = 20 * np.log10(thd_ratio)
    gain = (np.nanmax(Vp_valid) - np.nanmin(Vp_valid)) / pk_pk1

    return {
        "Vp1": Vp1_out,
        "Vp2": Vp2_out,
        "t": t,
        "THD": thd_ratio,
        "THD_dB": thd_db,
        "Gain": gain,
        "Vp1_op": np.nanmedian(Vp1_out),
        "Vp2_op": np.nanmedian(Vp2_out),
    }
