
import numpy as np
import matplotlib.pyplot as plt
from simulate_dc_coupled_chain import simulate_dc_coupled_chain

def optimize_dc_coupled_operating_point(
    triode1, triode2,
    pk_pk1=2.0,
    R_out1=1e6,
    R_out2=50000,
    Vg1_range=(-3.0, 0.0, 10),
    CCS1_range=(0.5, 3.0, 10),
    gain_threshold=0.1,
    sampling_rate=10000,
    f=1000,
    cycles=1.5
):
    Vg1_vals = np.linspace(*Vg1_range)
    CCS1_vals = np.linspace(*CCS1_range)

    THD_map = np.full((len(Vg1_vals), len(CCS1_vals)), np.nan)
    Gain_map = np.full_like(THD_map, np.nan)
    best_result = None
    best_thd = np.inf

    for i, Vg1_op in enumerate(Vg1_vals):
        for j, CCS1_current in enumerate(CCS1_vals):
            try:
                result = simulate_dc_coupled_chain(
                    triode1, triode2,
                    Vg1_op, CCS1_current,
                    R_out1, R_out2,
                    pk_pk1,
                    f=f,
                    sampling_rate=sampling_rate,
                    cycles=cycles
                )
                if np.isnan(result["THD"]) or result["Gain"] < gain_threshold:
                    continue
                THD_map[i, j] = result["THD"]
                Gain_map[i, j] = result["Gain"]

                if result["THD"] < best_thd:
                    best_thd = result["THD"]
                    best_result = {
                        "Vg1_op": Vg1_op,
                        "CCS1_current": CCS1_current,
                        **result
                    }
            except Exception as e:
                print(f"Error at Vg1={Vg1_op:.2f}, CCS1={CCS1_current:.2f}: {e}")
                continue

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        THD_map * 100,
        extent=[CCS1_vals[0], CCS1_vals[-1], Vg1_vals[0], Vg1_vals[-1]],
        origin="lower", aspect="auto", cmap="viridis"
    )
    plt.colorbar(im, ax=ax, label="THD (%)")
    ax.set_xlabel("CCS1 Current (mA)")
    ax.set_ylabel("Grid Bias Vg1_op (V)")
    ax.set_title(f"THD Heatmap — DC-Coupled Chain (pk-pk={pk_pk1}V)")

    if best_result:
        ax.plot(best_result["CCS1_current"], best_result["Vg1_op"], "ro", label="Best")
        ax.legend()
        print(f"✅ Best Vg1_op = {best_result['Vg1_op']:.2f} V, CCS1 = {best_result['CCS1_current']:.2f} mA")
        print(f"   Gain = {best_result['Gain']:.2f}, THD = {best_result['THD']*100:.2f}%%")
    else:
        print("⚠️ No valid solution found.")

    plt.tight_layout()
    plt.show()

    return best_result, Vg1_vals, CCS1_vals, THD_map, Gain_map
