
def optimize_joint(self, pk_pk1=2.0, f=1000, sampling_rate=10000):
    Vg1_ops = np.linspace(self.stage1.Vg_data.min() + 0.5, self.stage1.Vg_data.max() - 0.5, 20)
    CCS1_vals = np.linspace(0.2, 2.0, 20)  # in mA
    results = []

    for Vg1_op in Vg1_ops:
        for CCS1 in CCS1_vals:
            t = np.linspace(0, 1.5 / f, int(1.5 * sampling_rate))
            Vg1_in = Vg1_op + (pk_pk1 / 2) * np.sin(2 * np.pi * f * t)
            self.Vg1_in = Vg1_in  # Save for later use in evaluation
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

            # Get initial CCS2 estimate
            Vk2, Vp2_out = self.sweep_vk2(Vp1_out, CCS2_current=10)
            if Vp2_out is None or np.sum(~np.isnan(Vp2_out)) < 10:
                continue

            Ia_estimates = [self.interp2(vp2, vp1 - Vk2)
                            for vp1, vp2 in zip(Vp1_out, Vp2_out)
                            if not np.isnan(vp1) and not np.isnan(vp2)]
            CCS2 = np.nanmedian(Ia_estimates)
            if np.isnan(CCS2) or CCS2 < 0.5 or CCS2 > 40:
                continue

            # Evaluate and apply constraints
            result = self.evaluate_candidate(
                Vp1_out=Vp1_out,
                Vp2_out=Vp2_out,
                CCS1=CCS1,
                Vk2=Vk2,
                t=t,
                pk_pk1=pk_pk1,
                f=f,
                sampling_rate=sampling_rate
            )
            if result:
                result.update({
                    "Vg1_op": Vg1_op,
                    "CCS1": CCS1,
                    "Vk2": Vk2,
                    "CCS2": CCS2
                })
                results.append(result)

    return sorted(results, key=lambda x: x["THD"])
