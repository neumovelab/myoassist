# Author(s): Calder Robbins <robbins.cal@northeastern.edu>
import numpy as np
from scipy.interpolate import PchipInterpolator

class NPointSplineController:
    """Exoskeleton controller using an n-point spline for torque generation."""

    THRESHOLD = 0.1

    def __init__(self, n_points=4, dt=0.01, max_torque=10.0):
        self.dt = dt
        self.max_torque = max_torque
        
        # n-point parameters
        self.n_points = max(1, n_points)  # minimum of 1 point
        
        # Create dynamic parameter mapping for n-point spline
        self.param_keys = []
        self.param_map = {}

        # Create torque parameter keys (1 to n)
        for i in range(self.n_points):
            self.param_keys.append(f"torque_{i+1}")
        # Create time parameter keys (n+1 to n*2)
        for i in range(self.n_points):
            self.param_keys.append(f"time_{i+1}")
        # Create parameter map
        self.param_map = dict(zip(self.param_keys, range(len(self.param_keys))))
        
        # Initialize with default values across stance phase
        time_points = np.linspace(20, 80, self.n_points)  # Spread points between 20-80% for better initial distribution
        torque_points = np.ones(self.n_points)  # Default torque values of 1 (overridden to 0.5 x max_torque in train.py)
        self.spline_params = np.concatenate([torque_points, time_points/100])  # Normalize time to 0-1
        
        # Initialize torque spline
        self.torque_spline = self.precompute_torque_spline()
        
        # Tracking variables
        self.state = "SWING"
        self.stance_durations = []
        self.average_stance_duration = 0.6 # Seconds, estimated from preruns
        self.stride_counter = 0
        self.stance_duration_time = 0.0
        self.initial_override_done = False  # Track if initial override has been triggered

    def check_spline_validity(self): # Spline check triggered during pose validity check (see reflex_interface.py)
        """Check that timing parameters are valid for the spline profile."""
        # Get torque and time parameters
        torque_params = self.spline_params[:self.n_points]
        time_params = self.spline_params[self.n_points:]
        
        # Check if any torque values exceed normalized max_torque
        if np.any(torque_params > 1) or np.any(torque_params < 0.01):
            return False, "Torque values cannot exceed maximum torque"
        
        # Check if any torque values are negative
        if np.any(torque_params < 0):
            return False, "Torque values cannot be negative."
        
        # Check if any time points exceed 1 (100%)
        if np.any(time_params > 1):
            return False, "Time points cannot exceed 100 percent of stance phase."
        
        # Check if any time points are negative
        if np.any(time_params < 0):
            return False, "Time points cannot be negative."
            
        return True, "Valid spline parameters"

    def precompute_torque_spline(self):
        """Create a PCHIP spline function based on the current parameters."""
        return self._create_npoint_spline()

    def sort_spline_points(self, time_points, torque_points):
        """Sort spline points by time while maintaining torque pairing.
        
        Args:
            time_points (np.ndarray): Array of time points
            torque_points (np.ndarray): Array of corresponding torque values
            
        Returns:
            tuple: (sorted_time_points, sorted_torque_points)
        """
        # Create array of indices that would sort the time points
        sort_indices = np.argsort(time_points)
        
        # Use these indices to sort both arrays
        sorted_time_points = time_points[sort_indices]
        sorted_torque_points = torque_points[sort_indices]
        
        return sorted_time_points, sorted_torque_points

    def _create_npoint_spline(self):
        """Create spline using n-point implementation."""
        if len(self.spline_params) != self.n_points * 2:
            raise ValueError(f"Expected {self.n_points * 2} spline parameters, got {len(self.spline_params)}")
        
        # Split parameters into torque and time values
        torque_params = self.spline_params[:self.n_points] * self.max_torque
        normalized_time_params = self.spline_params[self.n_points:]
        
        # Calculate segment size
        segment_size = 1.0 / self.n_points
        
        # Denormalize each timing parameter to its specific segment
        time_params = np.zeros_like(normalized_time_params)
        for i in range(self.n_points):
            segment_min = i * segment_size
            segment_max = (i + 1) * segment_size
            # Map normalized parameter (0-1) to its specific segment
            time_params[i] = segment_min + (normalized_time_params[i] * (segment_max - segment_min))
        
        # Scale to 0-100% stance
        time_points = time_params * 100
        
        # Store control points for visualization
        self.control_time_points = time_points.copy()
        self.control_torque_points = torque_params.copy()
        
        # Add zero points at start and end
        pchip_time_points = np.concatenate([[0], time_points, [100]])
        pchip_torque_points = np.concatenate([[0], torque_params, [0]])
        
        # Store actual points used in the PCHIP interpolator
        self.pchip_time_points = pchip_time_points
        self.pchip_torque_points = pchip_torque_points
        
        # Create and return the PCHIP interpolator
        return PchipInterpolator(pchip_time_points, pchip_torque_points)

    def get_control_points(self, include_endpoints=False):
        """Return the control points used to create the spline.
        
        Args:
            include_endpoints (bool): Whether to include the added endpoints (0 and 100%)
                                     in the returned control points.
        
        Returns:
            tuple: (time_points, torque_points) - Arrays of control point times (0-100%) and torque values
        """
        if hasattr(self, 'control_time_points') and hasattr(self, 'control_torque_points'):
            if include_endpoints and hasattr(self, 'pchip_time_points'):
                return self.pchip_time_points, self.pchip_torque_points
            else:
                return self.control_time_points, self.control_torque_points
        else:
            return np.array([]), np.array([])

    def FSM(self, vgrf):
        """Determine stance or swing based on vGRF."""
        return "STANCE" if vgrf > self.THRESHOLD else "SWING"

    def update(self, vgrf, override=False, initial_percent=5):
        # Detect if we are in stance or swing (keep for stance duration tracking)
        new_state = self.FSM(vgrf)

        # Track stance durations and update averages
        if new_state == "STANCE":
            if self.state != "STANCE":
                self.stance_duration_time = 0.0
            self.stance_duration_time += self.dt
        else:  # SWING
            if self.state == "STANCE" and self.stance_duration_time > 0.035:
                self.stance_durations.append(self.stance_duration_time)
                self.stride_counter += 1
                if len(self.stance_durations) > 3:
                    self.stance_durations.pop(0)
                if len(self.stance_durations) == 3:
                    self.average_stance_duration = np.mean(self.stance_durations)

        # Calculate torque based on spline timing, regardless of state
        if override and not self.initial_override_done:
            current_percentage = initial_percent
            self.initial_override_done = True
        elif self.average_stance_duration is not None:
            current_percentage = np.clip((self.stance_duration_time / self.average_stance_duration) * 100, 0, 100)
        else:
            current_percentage = 0

        # Evaluate spline and clip torque
        torque = float(self.torque_spline(current_percentage))
        torque = np.clip(torque, 0, self.max_torque)

        # Update state
        self.state = new_state

        return torque

    def reset(self, param_vector):
        """Reset the controller parameters using the input vector."""
        expected_params = self.n_points * 2
        if len(param_vector) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters, got {len(param_vector)}")
        self.spline_params = param_vector.copy()
        
        # Recompute the torque spline with the updated parameters
        self.torque_spline = self.precompute_torque_spline() 