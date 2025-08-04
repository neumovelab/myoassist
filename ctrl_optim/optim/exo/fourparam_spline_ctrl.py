# Author(s): Calder Robbins <robbins.cal@northeastern.edu>
import numpy as np
from scipy.interpolate import PchipInterpolator

class FourParamSplineController:
    """Exoskeleton controller using a 4-parameter spline for torque generation."""
    
    THRESHOLD = 0.1

    spline_key = ['peak_torque', 'rise_time', 'peak_time', 'fall_time']
    spline_map = dict(zip(spline_key, range(len(spline_key))))


    def __init__(self, dt=0.01, max_torque=10.0, fixed_exo=False):
        self.dt = dt
        self.max_torque = max_torque
        self.fixed_exo = fixed_exo
        
        self.initial_params = {
            'peak_torque': 0.5,    # Initial normalized peak torque (0.5 = 50% of max_torque)
            'rise_time': 0.467,    # Initial rise time (normalized 0-1)
            'peak_time': 0.90,     # Initial peak time (normalized 0-1)
            'fall_time': 0.075      # Initial fall time (normalized 0-1)
        }
        
        self.peak_torque = None
        self.rise_time = None
        self.peak_time = None
        self.fall_time = None

        self.spline_params = np.array([
            self.initial_params['peak_torque'],
            self.initial_params['rise_time'],
            self.initial_params['peak_time'],
            self.initial_params['fall_time']
        ])
        
        self.torque_spline = self.precompute_torque_spline()
        
        # Tracking variables
        self.state = "SWING"
        self.stance_durations = []
        self.average_stance_duration = 0.6 # Seconds, estimated from preruns
        self.stride_counter = 0
        self.stance_duration_time = 0.0
        self.initial_override_done = False  # Track if initial override has been triggered

    def check_spline_validity(self):
        """Check that timing parameters are valid for the spline profile."""
        _, rise_time, peak_time, fall_time = self.spline_params
        
        # Scale timing parameters to percentages (0-100) for checks
        peak_time_pct = peak_time * 100
        rise_time_pct = rise_time * 100
        fall_time_pct = fall_time * 100

        if peak_time_pct > 100:
            return False, "Peak time exceeds 100% of stance phase."
        if peak_time_pct < 50:
            return False, "Peak time less than 50% of stance phase."
        if not (0 < rise_time_pct < peak_time_pct):
            return False, "Rise time must be between 0 and peak time."
        if not (0 < fall_time_pct < (100 - peak_time_pct)):
            return False, "Fall time must be within remaining stance after peak."
        return True, "Valid spline parameters"

    def precompute_torque_spline(self):
        """Create a PCHIP spline function based on the current parameters."""
        return self._create_legacy_spline()

    def _create_legacy_spline(self):
        """Create spline for original 4-point implementation"""
        # Scale peak torque by max_torque
        self.peak_torque = self.spline_params[0] * self.max_torque
        
        # Scale timing parameters from normalized (0-1) to percentages (0-100)
        self.rise_time = self.spline_params[1] * 100
        self.peak_time = self.spline_params[2] * 100
        self.fall_time = self.spline_params[3] * 100
        
        # Define phases and torque values as percentages of the stance phase
        phases = [0]  # Start at 0%
        torques = [0]

        # Rise phase - start of torque ramp up
        rise_start = max(0, self.peak_time - self.rise_time)
        if rise_start > 0:  # Only add rise point if it's not at 0
            phases.append(rise_start)
            torques.append(0)

        # Peak phase
        phases.append(self.peak_time)
        torques.append(self.peak_torque)

        # Fall phase - end of torque
        fall_end = min(100, self.peak_time + self.fall_time)
        if fall_end > phases[-1]:  # Only add if it's greater than the last point
            phases.append(fall_end)
            torques.append(0)

        # End of stance
        if phases[-1] < 100:  # Only add if we haven't reached 100 yet
            phases.append(100)
            torques.append(0)
        
        # Verify phases are strictly increasing
        for i in range(1, len(phases)):
            if phases[i] <= phases[i-1]:
                raise ValueError(f"Phase points must be strictly increasing. Got {phases}")
        
        # Store control points for visualization
        self.control_time_points = np.array(phases)
        self.control_torque_points = np.array(torques)
        
        # Create and return the PCHIP interpolator
        return PchipInterpolator(phases, torques)

    def get_control_points(self, include_endpoints=False):
        """Return the control points used to create the spline.
        
        Returns:
            tuple: (time_points, torque_points) - Arrays of control point times (0-100%) and torque values
        """
        if hasattr(self, 'control_time_points') and hasattr(self, 'control_torque_points'):
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
        if self.fixed_exo:
            self.spline_params = np.array([
                self.initial_params['peak_torque'],
                self.initial_params['rise_time'],
                self.initial_params['peak_time'],
                self.initial_params['fall_time']
            ])
        else:
            # Normal parameter handling
            expected_params = 4
            if len(param_vector) != expected_params:
                raise ValueError(f"Expected {expected_params} parameters, got {len(param_vector)}")
            self.spline_params = param_vector.copy()
        
        # Recompute the torque spline with the updated parameters
        self.torque_spline = self.precompute_torque_spline() 