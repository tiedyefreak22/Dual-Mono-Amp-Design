
import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Triode:
    def __init__(self, Vp_data, Ia_data, Vg_data, max_diss):
        self.Vp_data = Vp_data
        self.Ia_data = Ia_data
        self.Vg_data = Vg_data
        self.max_diss = max_diss

    def three_D_manifold(self):
        from scipy.interpolate import RBFInterpolator
        self.Vp_grid, self.Vg_grid = np.meshgrid(
            np.linspace(min(self.Vp_data), max(self.Vp_data), 100),
            np.linspace(min(self.Vg_data), max(self.Vg_data), 100)
        )

        # RBF interpolation for smooth surface
        rbf_interp = RBFInterpolator(
            np.column_stack((self.Vp_data, self.Vg_data)),
            self.Ia_data,
            smoothing=0.5
        )
        self.Ia_grid = rbf_interp(
            np.column_stack((self.Vp_grid.ravel(), self.Vg_grid.ravel()))
        ).reshape(self.Vp_grid.shape)

