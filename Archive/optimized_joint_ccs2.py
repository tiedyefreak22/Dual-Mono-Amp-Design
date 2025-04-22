
def optimize_joint(self, pk_pk1=2.0, f=1000, sampling_rate=10000):
    Vg1_ops = np.linspace(self.stage1.Vg_data.min() + 0.5, self.stage1.Vg_data.max() - 0.5, 20)
    CCS1_vals = np.linspace(0.2, 2.0, 20)  # in mA
    results = []

    for Vg1_op in Vg1_ops:
        for CCS1 in CCS1_vals:
            # --- Simulate Stage 1 ---
            t = np.linspace(0, 1.5 / f, int(1.5 * sampling_rate))
            Vg1_in = Vg1_op + (pk_pk1 / 2) * np.sin(2 * np.pi * f * t)
            Vp1_out = []
            last_vp1 = 150

            for Vg1 in Vg1_in:
                def res1(Vp):
                    Ia = self.interp1(Vp, Vg1)
                    return 1e6 if Ia is None or np.isnan(Ia) else Ia - (CCS1 - Vp / self.R_out1)
                try:
                    sol1 = root_scalar(res1, bracket=[last_vp1 - 10, last_vp1 + 10], method='brentq')
                    last_vp1 = sol1.root if sol1.converged else last_vp1
                except:
                    try:
                        sol1 = root_scalar(res1, bracket=[0, 400], method='brentq')
                        last_vp1 = sol1.root if sol1.converged else last_vp1
                    except:
                        sol1 = None
                Vp1_out.append(sol1.root if sol1 and sol1.converged else np.nan)

            Vp1_out = np.array(Vp1_out)
            if np.all(np.isnan(Vp1_out)):
                continue

            # Use median output swing to determine CCS2 more accurately
            Vk2, Vp2_out = self.sweep_vk2(Vp1_out, CCS2_current=10)  # start with dummy value
            if Vp2_out is None or np.sum(~np.isnan(Vp2_out)) < 10:
                continue

            Ia_estimates = [self.interp2(vp2, vp1 - Vk2)
                            for vp1, vp2 in zip(Vp1_out, Vp2_out)
                            if not np.isnan(vp1) and not np.isnan(vp2)]
            CCS2 = np.nanmedian(Ia_estimates)
            if np.isnan(CCS2) or CCS2 < 0.5 or CCS2 > 40:
                continue

            # --- Evaluate performance ---
            Vp2_out = np.array(Vp2_out)
            valid = ~np.isnan(Vp2_out)
            Vp_valid = Vp2_out[valid]

            t_trimmed = t[:len(Vp2_out)]
            t_valid = t_trimmed[valid]

            yf = rfft(Vp_valid - np.mean(Vp_valid))
            freqs = rfftfreq(len(Vp_valid), 1 / sampling_rate)
            harmonics = np.abs(yf)
            fundamental = harmonics[1] if len(harmonics) > 1 else 1e-6
            harmonic_power = np.sum(harmonics[2:] ** 2)
            thd = np.sqrt(harmonic_power) / fundamental
            gain = (np.nanmax(Vp_valid) - np.nanmin(Vp_valid)) / pk_pk1
            margin = np.min([np.abs(Vp1 - Vk2 - self.stage2.Vg_data.min()) for Vp1 in Vp1_out if not np.isnan(Vp1)])

            results.append({
                "Vg1_op": Vg1_op,
                "CCS1": CCS1,
                "Vk2": Vk2,
                "CCS2": CCS2,
                "Gain": gain,
                "THD": thd,
                "Margin": margin,
                "Vp1_op": np.nanmedian(Vp1_out),
                "Vp2_op": np.nanmedian(Vp2_out)
            })

    results = sorted(results, key=lambda x: x["THD"])
    return results
