
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
            gain = (np.nanmax(Vp_valid) - np.nanmin(Vp_valid)) / pk_pk
            gain_list.append(gain)
            thd_list.append(thd_ratio)
        except Exception as e:
            gain_list.append(np.nan)
            thd_list.append(np.nan)
            print(f"pk_pk={pk_pk}: {e}")

    pk_pks = np.array(pk_pks)
    gain_list = np.array(gain_list)
    thd_list = np.array(thd_list)

    # Determine best linearity region
    valid = ~np.isnan(thd_list) & ~np.isnan(gain_list)
    gain_thresh = 0.1  # minimum usable gain
    idx_best = np.argmin(np.where(gain_list > gain_thresh, thd_list, np.inf))

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
    ax2.plot(pk_pks, thd_list * 100, 's--', color=color, label="THD")
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight best linearity region
    if valid.any():
        ax1.axvline(pk_pks[idx_best], color='green', linestyle='--', alpha=0.7)
        ax2.axvline(pk_pks[idx_best], color='green', linestyle='--', alpha=0.7)
        ax2.annotate("Best linearity", xy=(pk_pks[idx_best], thd_list[idx_best] * 100),
                     xytext=(pk_pks[idx_best] + 0.2, thd_list[idx_best] * 100 + 2),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     color='green')

    fig.tight_layout()
    plt.title("Gain and THD vs Input Swing (with Best Linearity)")
    plt.show()

    return pk_pks, gain_list, thd_list, pk_pks[idx_best], gain_list[idx_best], thd_list[idx_best]
