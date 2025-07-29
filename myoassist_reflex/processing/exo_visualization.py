"""
Exoskeleton Control Visualization Module

This module provides real-time visualization of exoskeleton control parameters
during simulation, including stance phase tracking, applied torque, and torque profiles.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class ExoVisualizer:
    """Real-time exoskeleton control visualizer."""
    
    def __init__(self, frame_width: int = 1920, frame_height: int = 1072):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Colors (BGR format for OpenCV)
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'orange': (0, 165, 255)
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale_large = 1.2
        self.font_scale_medium = 1.0
        self.font_scale_small = 0.8
        self.thickness = 2
        
        # Layout settings
        self.margin = 30
        self.line_height = 45
        
    def create_overlay(self, env, current_torque_r: float, current_torque_l: float,
                      stance_percent_r: float, stance_percent_l: float,
                      state_r: str, state_l: str) -> np.ndarray:
        """Create a real-time overlay showing exoskeleton control information."""
        
        # Create overlay canvas (transparent background)
        overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Header
        cv2.putText(overlay, "EXOSKELETON CONTROL VISUALIZATION", 
                    (self.margin, self.margin), self.font, self.font_scale_large, 
                    self.colors['black'], self.thickness)
        
        # Right leg info
        y_pos = self.margin + 70
        cv2.putText(overlay, f"RIGHT LEG:", (self.margin, y_pos), self.font, 
                    self.font_scale_medium, self.colors['black'], self.thickness)
        y_pos += self.line_height
        cv2.putText(overlay, f"  State: {state_r}", (self.margin, y_pos), self.font, 
                    self.font_scale_medium, 
                    self.colors['green'] if state_r == "STANCE" else self.colors['red'], 
                    self.thickness)
        y_pos += self.line_height
        cv2.putText(overlay, f"  Stance Phase: {stance_percent_r:.1f}%", 
                    (self.margin, y_pos), self.font, self.font_scale_medium, 
                    self.colors['black'], self.thickness)
        y_pos += self.line_height
        cv2.putText(overlay, f"  Applied Torque: {current_torque_r:.2f} Nm", 
                    (self.margin, y_pos), self.font, self.font_scale_medium, 
                    self.colors['orange'], self.thickness)
        
        # Left leg info
        y_pos += self.line_height + 15
        cv2.putText(overlay, f"LEFT LEG:", (self.margin, y_pos), self.font, 
                    self.font_scale_medium, self.colors['black'], self.thickness)
        y_pos += self.line_height
        cv2.putText(overlay, f"  State: {state_l}", (self.margin, y_pos), self.font, 
                    self.font_scale_medium, 
                    self.colors['green'] if state_l == "STANCE" else self.colors['red'], 
                    self.thickness)
        y_pos += self.line_height
        cv2.putText(overlay, f"  Stance Phase: {stance_percent_l:.1f}%", 
                    (self.margin, y_pos), self.font, self.font_scale_medium, 
                    self.colors['black'], self.thickness)
        y_pos += self.line_height
        cv2.putText(overlay, f"  Applied Torque: {current_torque_l:.2f} Nm", 
                    (self.margin, y_pos), self.font, self.font_scale_medium, 
                    self.colors['orange'], self.thickness)
        
        # Torque profile visualization (right side of screen)
        if env.exo_bool and env.ExoCtrl_R is not None:
            self._draw_torque_profile(overlay, env, stance_percent_r)
        
        return overlay
    
    def _draw_torque_profile(self, overlay: np.ndarray, env, stance_percent_r: float):
        """Draw the torque profile graph."""
        profile_width = 500  # Larger profile
        profile_height = 250  # Larger profile
        profile_x = self.frame_width - profile_width - self.margin
        profile_y = self.margin + 70
        
        # Create torque profile plot
        x_vals = np.linspace(0, 100, 101)
        try:
            if env.use_4param_spline:
                torque_vals = np.array([env.ExoCtrl_R.torque_spline(v) for v in x_vals])
            else:
                torque_vals = np.array([env.ExoCtrl_R.torque_spline(v) for v in x_vals])
        except:
            torque_vals = np.zeros_like(x_vals)
        
        # Normalize for visualization
        max_torque = max(1, np.max(torque_vals))
        normalized_torque = (torque_vals / max_torque) * profile_height * 0.8
        
        # Draw profile background
        cv2.rectangle(overlay, (profile_x, profile_y), 
                     (profile_x + profile_width, profile_y + profile_height), 
                     self.colors['black'], 3)
        
        # Draw torque curve (cleaner without grid lines)
        for i in range(len(x_vals) - 1):
            x1 = int(profile_x + (x_vals[i] / 100) * profile_width)
            y1 = int(profile_y + profile_height - normalized_torque[i])
            x2 = int(profile_x + (x_vals[i+1] / 100) * profile_width)
            y2 = int(profile_y + profile_height - normalized_torque[i+1])
            cv2.line(overlay, (x1, y1), (x2, y2), self.colors['blue'], 3)
        
        # Draw current position indicator
        if stance_percent_r > 0:
            current_x = int(profile_x + (stance_percent_r / 100) * profile_width)
            current_y = int(profile_y + profile_height - normalized_torque[int(stance_percent_r)])
            cv2.circle(overlay, (current_x, current_y), 12, self.colors['red'], -1)
            cv2.circle(overlay, (current_x, current_y), 12, self.colors['black'], 2)
        
        # Labels
        cv2.putText(overlay, "TORQUE PROFILE", (profile_x, profile_y - 15), 
                    self.font, self.font_scale_medium, self.colors['black'], self.thickness)
        cv2.putText(overlay, "0%", (profile_x, profile_y + profile_height + 30), 
                    self.font, self.font_scale_small, self.colors['black'], 1)
        cv2.putText(overlay, "100%", (profile_x + profile_width - 40, profile_y + profile_height + 30), 
                    self.font, self.font_scale_small, self.colors['black'], 1)
        
        # Add max torque label
        cv2.putText(overlay, f"Max: {max_torque:.1f} Nm", 
                    (profile_x, profile_y - 40), self.font, self.font_scale_small, 
                    self.colors['black'], 1)
    
    def overlay_on_frame(self, frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Blend overlay onto frame without dimming."""
        # Resize overlay to match frame dimensions
        overlay_resized = cv2.resize(overlay, (frame.shape[1], frame.shape[0]))
        
        # Create mask for non-black pixels in overlay
        mask = np.any(overlay_resized > 0, axis=2)
        
        # Apply overlay only where there's content
        result = frame.copy()
        result[mask] = overlay_resized[mask]
        
        return result
    
    def get_exo_control_data(self, env) -> Dict[str, float]:
        """Extract current exoskeleton control data from environment."""
        current_torque_r = 0.0
        current_torque_l = 0.0
        stance_percent_r = 0.0
        stance_percent_l = 0.0
        state_r = "SWING"
        state_l = "SWING"
        
        if env.exo_bool and env.ExoCtrl_R is not None:
            # Get right leg data
            if hasattr(env.ExoCtrl_R, 'state'):
                state_r = env.ExoCtrl_R.state
            if hasattr(env.ExoCtrl_R, 'stance_duration_time') and hasattr(env.ExoCtrl_R, 'average_stance_duration'):
                if env.ExoCtrl_R.average_stance_duration > 0:
                    stance_percent_r = np.clip((env.ExoCtrl_R.stance_duration_time / env.ExoCtrl_R.average_stance_duration) * 100, 0, 100)
            
            # Get left leg data
            if hasattr(env.ExoCtrl_L, 'state'):
                state_l = env.ExoCtrl_L.state
            if hasattr(env.ExoCtrl_L, 'stance_duration_time') and hasattr(env.ExoCtrl_L, 'average_stance_duration'):
                if env.ExoCtrl_L.average_stance_duration > 0:
                    stance_percent_l = np.clip((env.ExoCtrl_L.stance_duration_time / env.ExoCtrl_L.average_stance_duration) * 100, 0, 100)
            
            # Get actual applied torques directly from simulation actuators
            try:
                # Access the actual applied torque from the simulation
                if 'Exo_R' in env.torque_dict and len(env.torque_dict['Exo_R']) > 0:
                    exo_r_idx = env.torque_dict['Exo_R'][0]
                    # Convert control signal back to torque using actuator gain
                    control_signal = env.env.sim.data.ctrl[exo_r_idx]
                    actuator_gain = env.env.sim.model.actuator('Exo_R').gainprm[0]
                    current_torque_r = abs(control_signal * actuator_gain)
                
                if 'Exo_L' in env.torque_dict and len(env.torque_dict['Exo_L']) > 0:
                    exo_l_idx = env.torque_dict['Exo_L'][0]
                    # Convert control signal back to torque using actuator gain
                    control_signal = env.env.sim.data.ctrl[exo_l_idx]
                    actuator_gain = env.env.sim.model.actuator('Exo_L').gainprm[0]
                    current_torque_l = abs(control_signal * actuator_gain)
                    
            except Exception as e:
                # Fallback to controller values if direct access fails
                try:
                    if hasattr(env.ExoCtrl_R, 'last_torque'):
                        current_torque_r = env.ExoCtrl_R.last_torque
                    if hasattr(env.ExoCtrl_L, 'last_torque'):
                        current_torque_l = env.ExoCtrl_L.last_torque
                except:
                    pass
        
        return {
            'torque_r': current_torque_r,
            'torque_l': current_torque_l,
            'stance_percent_r': stance_percent_r,
            'stance_percent_l': stance_percent_l,
            'state_r': state_r,
            'state_l': state_l
        } 