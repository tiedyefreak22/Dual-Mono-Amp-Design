
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import root_scalar
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

class TriodeChain:
    def __init__(self, stage1, stage2, R_out1=1e6, R_out2=50000):
        self.stage1 = stage1
        self.stage2 = stage2
        self.R_out1 = R_out1
        self.R_out2 = R_out2
        self.stage1.three_D_manifold()
        self.stage2.three_D_manifold()
        self.interp1 = LinearNDInterpolator(
            np.column_stack((stage1.Vp_grid.ravel(), stage1.Vg_grid.ravel())),
            stage1.Ia_grid.ravel()
        )
        self.interp2 = LinearNDInterpolator(
            np.column_stack((stage2.Vp_grid.ravel(), stage2.Vg_grid.ravel())),
            stage2.Ia_grid.ravel()
        )

    def simulate(self, Vg1_op, CCS1_current, pk_pk1=2.0, f=1000, cycles=1.5, sampling_rate=10000):
        t = np.linspace(0, cycles / f, int(cycles * sampling_rate))
        Vg1_in = Vg1_op + (pk_pk1 / 2) * np.sin(2 * np.pi * f * t)
        Vp1_out, Vp2_out = [], []
        last_vp1, last_vp2 = 150, 150

        for Vg1 in Vg1_in:
            def res1(Vp): Ia = self.interp1(Vp, Vg1); return 1e6 if Ia is None or np.isnan(Ia) else Ia - (CCS1_current - Vp / self.R_out1)
            try:
                sol1 = root_scalar(res1, bracket=[last_vp1 - 10, last_vp1 + 10], method='brentq')
                last_vp1 = sol1.root if sol1.converged else last_vp1
            except: sol1 = root_scalar(res1, bracket=[0, 400], method='brentq') if res1(0)*res1(400)<0 else None
            Vp1_out.append(sol1.root if sol1 and sol1.converged else np.nan)

        Vp1_out = np.array(Vp1_out)
        Vg2_in = Vp1_out
        CCS2_current = np.nanmedian(Vp1_out / self.R_out1)

        for Vg2 in Vg2_in:
            def res2(Vp): Ia = self.interp2(Vp, Vg2); return 1e6 if Ia is None or np.isnan(Ia) else Ia - (CCS2_current - Vp / self.R_out2)
            try:
                sol2 = root_scalar(res2, bracket=[last_vp2 - 10, last_vp2 + 10], method='brentq')
                last_vp2 = sol2.root if sol2.converged else last_vp2
            except: sol2 = root_scalar(res2, bracket=[0, 400], method='brentq') if res2(0)*res2(400)<0 else None
            Vp2_out.append(sol2.root if sol2 and sol2.converged else np.nan)

        Vp2_out = np.array(Vp2_out)
        valid = ~np.isnan(Vp2_out)
        if np.sum(valid) < 10: return None

        Vp_valid = Vp2_out[valid]
        t_valid = t[valid]
        yf = rfft(Vp_valid - np.mean(Vp_valid))
        freqs = rfftfreq(len(Vp_valid), 1 / sampling_rate)
        harmonics = np.abs(yf)
        fundamental = harmonics[1] if len(harmonics) > 1 else 1e-6
        harmonic_power = np.sum(harmonics[2:] ** 2)
        thd = np.sqrt(harmonic_power) / fundamental
        gain = (np.nanmax(Vp_valid) - np.nanmin(Vp_valid)) / pk_pk1

        return {
            "t": t_valid,
            "Vp1": Vp1_out,
            "Vp2": Vp2_out,
            "THD": thd,
            "Gain": gain,
            "Vp1_op": np.nanmedian(Vp1_out),
            "Vp2_op": np.nanmedian(Vp2_out),
        }

    def optimize(self, pk_pk1=2.0, Vg1_range=(-3.0, 0.0, 10), CCS1_range=(0.5, 3.0, 10),
                 gain_threshold=0.1, f=1000, cycles=1.5, sampling_rate=10000):
        Vg1_vals = np.linspace(*Vg1_range)
        CCS1_vals = np.linspace(*CCS1_range)
        best_result, best_thd = None, np.inf
        THD_map, Gain_map = np.full((len(Vg1_vals), len(CCS1_vals)), np.nan), np.full((len(Vg1_vals), len(CCS1_vals)), np.nan)

        for i, Vg1_op in enumerate(Vg1_vals):
            for j, CCS1 in enumerate(CCS1_vals):
                result = self.simulate(Vg1_op, CCS1, pk_pk1, f, cycles, sampling_rate)
                if not result or result["Gain"] < gain_threshold: continue
                THD_map[i, j], Gain_map[i, j] = result["THD"], result["Gain"]
                if result["THD"] < best_thd:
                    best_result = dict(result, Vg1_op=Vg1_op, CCS1=CCS1)
                    best_thd = result["THD"]

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(THD_map * 100, extent=[CCS1_vals[0], CCS1_vals[-1], Vg1_vals[0], Vg1_vals[-1]],
                       origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax, label="THD (%)")
        ax.set_xlabel("CCS1 Current (mA)")
        ax.set_ylabel("Grid Bias Vg1_op (V)")
        ax.set_title("THD Heatmap — DC-Coupled Chain")
        if best_result:
            ax.plot(best_result["CCS1"], best_result["Vg1_op"], 'ro', label="Best")
            ax.legend()
            print(f"✅ Best Vg1_op={best_result['Vg1_op']:.2f}V, CCS1={best_result['CCS1']:.2f}mA, Gain={best_result['Gain']:.2f}, THD={best_result['THD']*100:.2f}%")
        plt.tight_layout()
        plt.show()
        return best_result, Vg1_vals, CCS1_vals, THD_map, Gain_map
