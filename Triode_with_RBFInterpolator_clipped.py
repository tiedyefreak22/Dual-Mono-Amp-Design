
import numpy as np
from scipy.optimize import root_scalar
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

class Triode:
    def __init__(self, Vp_data, Ia_data, Vg_data, max_diss):
        self.Vp_data = Vp_data
        self.Ia_data = Ia_data
        self.Vg_data = Vg_data
        self.max_diss = max_diss

    def three_D_manifold(self):
        from scipy.interpolate import RBFInterpolator
        from scipy.spatial import Delaunay

        # Generate interpolation grid
        self.Vp_grid, self.Vg_grid = np.meshgrid(
            np.linspace(min(self.Vp_data), max(self.Vp_data), 100),
            np.linspace(min(self.Vg_data), max(self.Vg_data), 100)
        )
        grid_points = np.column_stack((self.Vp_grid.ravel(), self.Vg_grid.ravel()))

        # Fit RBF interpolator
        rbf_interp = RBFInterpolator(
            np.column_stack((self.Vp_data, self.Vg_data)),
            self.Ia_data,
            smoothing=0.5
        )
        Ia_rbf = rbf_interp(grid_points)

        # Mask extrapolated values using Delaunay triangulation
        hull = Delaunay(np.column_stack((self.Vp_data, self.Vg_data)))
        mask = hull.find_simplex(grid_points) >= 0
        Ia_rbf[~mask] = np.nan  # set extrapolated values to NaN

        # Reshape to grid
        self.Ia_grid = Ia_rbf.reshape(self.Vp_grid.shape)

