
import numpy as np
import matplotlib.pyplot as plt
from simulate_thd import simulate_thd

def gain_and_thd_vs_swing(triode, Vg_op, swing_range, R_out, CCS_current, f=1000, sampling_rate=100000):
    pk_pks = np.linspace(*swing_range)
    thd_list = []
    gain_list = []

    for pk_pk in pk_pks:
        try:
            thd_ratio, thd_db, t, Vp_valid = simulate_thd(
                triode, Vg_op, pk_pk, R_out, CCS_current,
                f=f, sampling_rate=sampling_rate
            )
            # Estimate gain as peak-to-peak output / peak-to-peak input
            gain = (np.nanmax(Vp_valid) - np.nanmin(Vp_valid)) / pk_pk
            gain_list.append(gain)
            thd_list.append(thd_ratio)
        except Exception as e:
            gain_list.append(np.nan)
            thd_list.append(np.nan)
            print(f"pk_pk={pk_pk}: {e}")

    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel("Input Swing (V pk-pk)")
    ax1.set_ylabel("Gain (V/V)", color=color)
    ax1.plot(pk_pks, gain_list, 'o-', color=color, label="Gain")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel("THD (%)", color=color)
    ax2.plot(pk_pks, np.array(thd_list) * 100, 's--', color=color, label="THD")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Gain and THD vs Input Swing")
    plt.show()

    return pk_pks, gain_list, thd_list
