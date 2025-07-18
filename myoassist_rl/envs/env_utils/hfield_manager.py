from myosuite.physics.sim_scene import SimScene
import numpy as np
import math
class HfieldManager:
    def __init__(self, sim:SimScene, hfield_name:str, np_random:np.random.RandomState):
        self._hfield = sim.model.hfield(hfield_name)
        self._hfield_pos = sim.model.geom(hfield_name).pos
        self._hfield_size = sim.model.geom(hfield_name).size
        self.np_random = np_random

    def set_hfield(self, type:str="dev"):
        if type == "flat":
            pass
        elif type == "random":
            self._create_random_hfield()
        elif type == "sinusoidal":
            self._create_sinusoidal_hfield()
        elif type == "harmonic_sinusoidal":
            self._create_harmonic_sinusoidal_hfield()
        elif type == "uphill":
            self._create_slope_hfield(0.3)
        elif type == "downhill":
            raise NotImplementedError("Downhill terrain is not implemented yet")
            self._create_slope_hfield(-0.3)# not working yet
        elif type == "dev":
            self._create_slope_hfield(0.3)

    def _make_safe_zone(self, hfield_data):
        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)  # Ensure nrow and ncol are integers


        # print(f"{nrow=}, {ncol=} {self._hfield_size=}")


        safezone_radius = 3.0
        tile_size_row = 2 * self._hfield.size[1] / nrow# size is radius
        tile_size_col = 2 * self._hfield.size[0] / ncol
        center_index = int((self._hfield_size[0] - self._hfield_pos[0]) / tile_size_col)
        tile_num_safezone = math.ceil(safezone_radius / tile_size_row)
        tile_num_safezone_col = math.ceil(safezone_radius / tile_size_col)

        # Create a smooth mask that is 0 at the center and 1 at the edge of the safe zone.
        center_row = nrow // 2
        center_col = center_index
        row_start = center_row - tile_num_safezone
        row_end = center_row + tile_num_safezone
        col_start = center_col - tile_num_safezone_col
        col_end = center_col + tile_num_safezone_col

        print(f"{center_index=} {center_row=} {center_col=}")
        print(f"{row_start=}, {row_end=} {col_start=}, {col_end=}")

        # Generate grid for the safe zone
        safezone_rows = np.arange(row_start, row_end)
        safezone_cols = np.arange(col_start, col_end)
        safezone_row_grid, safezone_col_grid = np.meshgrid(safezone_rows, safezone_cols, indexing='ij')

        # Calculate distance from the center for each point in the safe zone
        dist_from_center = np.sqrt(
            ((safezone_row_grid - center_row) * tile_size_row) ** 4 +
            ((safezone_col_grid - center_col) * tile_size_col) ** 4
        )

        # Normalize distance to [0, 1] within the safe zone radius
        mask = np.clip(dist_from_center / safezone_radius, 0, 1)

        # Set mask values to 0 where mask is less than or equal to 0.9
        # mask[mask <= 0.9] = 0

        # Multiply the original hfield data in the safe zone by the mask
        hfield_data[row_start:row_end, col_start:col_end] *= mask
        return hfield_data

    def _create_random_hfield(self):

        amplitude = 0.3
        

        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)  # Ensure nrow and ncol are integers
        self._hfield.data[:] = self._make_safe_zone(self.np_random.uniform(low=0, high=amplitude, size=(nrow, ncol)))


    def _create_sinusoidal_hfield(self):

        row_period = 20
        col_period = 20
        amplitude = 0.2

        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)  # Ensure nrow and ncol are integers

        row_idx = np.arange(nrow)
        col_idx = np.arange(ncol)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing='ij')

        freq_row = 2 * np.pi / row_period
        freq_col = 2 * np.pi / col_period

        # Compute the sinusoidal height values
        # Compute the sinusoidal height values and shift so that minimum is at least 0
        hfield_data = amplitude * (
            np.sin(freq_row * row_grid) + np.sin(freq_col * col_grid)
        )
        min_val = np.min(hfield_data)
        if min_val < 0:
            # Shift the entire hfield_data so that minimum is 0
            hfield_data = hfield_data - min_val

        # self._hfield.data[:] = hfield_data

        self._hfield.data[:] = self._make_safe_zone(hfield_data)

    def _create_harmonic_sinusoidal_hfield(self):

        row_period = 20
        col_period = 20
        amplitude = 0.2

        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)  # Ensure nrow and ncol are integers

        row_idx = np.arange(nrow)
        col_idx = np.arange(ncol)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing='ij')

        freq_row = 2 * np.pi / row_period
        freq_col = 2 * np.pi / col_period

        row_params = [(20, 0.2), (60, 0.05)]
        col_params = [(8, 0.1), (40, 0.5), (80, 1.0)]

        hfield_data = np.zeros_like(row_grid, dtype=np.float32)
        # Add row-direction sinusoids
        for period, amp in row_params:
            freq_row = 2 * np.pi / period
            hfield_data += amp * np.sin(freq_row * row_grid)
        # Add col-direction sinusoids
        for period, amp in col_params:
            freq_col = 2 * np.pi / period
            hfield_data += amp * np.sin(freq_col * col_grid)
        min_val = np.min(hfield_data)
        if min_val < 0:
            # Shift the entire hfield_data so that minimum is 0
            hfield_data = hfield_data - min_val

        # self._hfield.data[:] = hfield_data

        self._hfield.data[:] = self._make_safe_zone(hfield_data)

    
    def _create_slope_hfield(self, slope:float):
        nrow, ncol = int(self._hfield.nrow), int(self._hfield.ncol)  # Ensure nrow and ncol are integers

        tile_size_row = 2 * self._hfield.size[1] / nrow# size is radius
        tile_size_col = 2 * self._hfield.size[0] / ncol
        center_index = int((self._hfield_size[0] - self._hfield_pos[0]) / tile_size_col) + 5

        row_idx = np.arange(nrow)
        col_idx = np.arange(ncol)
        row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing='ij')

        hfield_data = np.zeros_like(row_grid, dtype=np.float32)

        # Only add slope for columns where (col_grid - center_index) > 0, otherwise add 0.
        # This ensures that for col_grid <= center_index, the value added is 0.
        hfield_data += np.where(
            (col_grid - center_index) > 0,
            (col_grid - center_index) * slope * tile_size_row,
            0,
            
        )

        self._hfield.data[:] = hfield_data
    