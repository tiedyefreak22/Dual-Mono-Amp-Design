
def evaluate_candidate(self, Vp1_out, Vp2_out, CCS1, Vk2, t, pk_pk1, f, sampling_rate):
    from scipy.fft import rfft, rfftfreq
    import numpy as np

    Vp1_out = np.array(Vp1_out)
    Vp2_out = np.array(Vp2_out)

    if np.all(np.isnan(Vp2_out)):
        return None

    # Validity checks
    valid = ~np.isnan(Vp2_out)
    if np.sum(valid) < 100:
        return None

    t_trimmed = t[:len(Vp2_out)]
    t_valid = t_trimmed[valid]
    Vp_valid = Vp2_out[valid]

    # FFT
    yf = rfft(Vp_valid - np.mean(Vp_valid))
    freqs = rfftfreq(len(Vp_valid), 1 / sampling_rate)
    harmonics = np.abs(yf)
    fundamental = harmonics[1] if len(harmonics) > 1 else 1e-6
    harmonic_power = np.sum(harmonics[2:] ** 2)
    thd = np.sqrt(harmonic_power) / fundamental
    gain = (np.nanmax(Vp_valid) - np.nanmin(Vp_valid)) / pk_pk1

    # Estimate Ia1, Ia2 from interp
    Ia1 = np.array([self.interp1(vp1, Vg1) if not np.isnan(vp1) else np.nan for vp1, Vg1 in zip(Vp1_out, self.Vg1_in)])
    Ia2 = np.array([self.interp2(vp2, vp1 - Vk2) if not np.isnan(vp2) and not np.isnan(vp1) else np.nan for vp1, vp2 in zip(Vp1_out, Vp2_out)])

    # Plate dissipation
    Pdiss1 = np.nanmedian(Ia1 * Vp1_out)
    Pdiss2 = np.nanmedian(Ia2 * Vp2_out)

    if Pdiss1 > 1.0 or Pdiss2 > 4.0:
        return None

    # Grid overdrive protection
    Vg2 = Vp1_out - Vk2
    if np.any(Vg2 > 0.2):
        return None

    # Output drive into load (e.g., 600Ω headphones)
    Rload = 600
    Iload = Vp_valid / Rload
    if np.nanmax(np.abs(Iload)) > 0.05:  # 50mA max
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
