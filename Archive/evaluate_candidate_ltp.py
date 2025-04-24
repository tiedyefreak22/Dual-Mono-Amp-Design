
def evaluate_candidate(self, Vp1_out, Vp2_out, CCS1, Vk2, t, pk_pk1, f, sampling_rate):
    from scipy.fft import rfft, rfftfreq
    import numpy as np

    Vp1_out = np.array(Vp1_out)
    Vp2_out = np.array(Vp2_out)

    if np.all(np.isnan(Vp2_out)):
        print("[Reject] All Vp2_out values are NaN.")
        return None

    valid = ~np.isnan(Vp2_out)
    if np.sum(valid) < 100:
        print("[Reject] Too few valid Vp2 points.")
        return None

    t_trimmed = t[:len(Vp2_out)]
    Vp_valid = Vp2_out[valid]

    yf = rfft(Vp_valid - np.mean(Vp_valid))
    freqs = rfftfreq(len(Vp_valid), 1 / sampling_rate)
    harmonics = np.abs(yf)
    fundamental = harmonics[1] if len(harmonics) > 1 else 1e-6
    harmonic_power = np.sum(harmonics[2:] ** 2)
    thd = np.sqrt(harmonic_power) / fundamental
    gain = (np.nanmax(Vp_valid) - np.nanmin(Vp_valid)) / pk_pk1

    Ia1 = np.array([self.interp1(vp1, Vg1) if not np.isnan(vp1) else np.nan for vp1, Vg1 in zip(Vp1_out, self.Vg1_in)])
    Ia2 = np.array([self.interp2(vp2, vp1 - Vk2) if not np.isnan(vp2) and not np.isnan(vp1) else np.nan for vp1, vp2 in zip(Vp1_out, Vp2_out)])

    Pdiss1 = np.nanmedian((Ia1 / 1000) * Vp1_out)
    Pdiss2 = np.nanmedian((Ia2 / 1000) * Vp2_out)

    if Pdiss1 > 1.0 or Pdiss2 > 4.0:
        print(f"[Reject] Pdiss1={Pdiss1:.2f}W Pdiss2={Pdiss2:.2f}W exceeds limits.")
        return None

    Vg2 = Vp1_out - Vk2
    if np.nanmax(Vg2) > 0.5:
        print(f"[Reject] Vg2 excursion too high: max={np.nanmax(Vg2):.2f}V")
        return None

    Rload = 600
    Iload = Vp_valid / Rload
    if np.nanmax(np.abs(Iload)) > 0.05:
        print(f"[Reject] Output current exceeded 50mA: Max Iout = {np.nanmax(np.abs(Iload)):.3f}A")
        return None

    margin = np.min([np.abs(Vp1 - Vk2 - self.stage2.Vg_data.min()) for Vp1 in Vp1_out if not np.isnan(Vp1)])

    return {
        "Gain": gain,
        "THD": thd,
        "Margin": margin,
        "Pdiss1": Pdiss1,
        "Pdiss2": Pdiss2,
        "Vp1_op": np.nanmedian(Vp1_out),
        "Vp2_op": np.nanmedian(Vp2_out)
    }
