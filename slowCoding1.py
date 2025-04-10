#Þetta file er basically bara drasl.

# Required Libraries: pip install krpc pygad numpy matplotlib
import krpc
import time
import math
import pygad
import numpy as np
import matplotlib.pyplot as plt # Needed for ga_instance.plot_fitness()

# --- Configuration ---
KSP_CONNECTION_NAME = 'GA_PID_Tuner_Heading_VSI'
TARGET_HEADING = 45      # Target heading in degrees
TARGET_VERTICAL_SPEED = 1.0 # Target vertical speed in m/s (e.g., 1.0 for a gentle climb)
SIMULATION_DURATION = 15 # How many seconds to test each set of PID gains (Increased slightly for VSI stability)
CONTROL_UPDATE_INTERVAL = 0.05 # How often to update PID and controls (seconds) - slightly faster updates might help
INITIAL_THROTTLE = 1.0   # Initial throttle setting for the test flight

# --- Fitness Function Weights (TUNE THESE!) ---
# How much to prioritize heading accuracy vs vertical speed accuracy
# Higher weight means that aspect is more important for the fitness score.
HEADING_WEIGHT = 1.0
VSI_WEIGHT = 0.5 # Start with VSI being less critical than heading, adjust as needed
# Penalty for diving significantly below the target VSI
DIVING_PENALTY_MULTIPLIER = 5.0 # Increase penalty if average VSI is well below target
VSI_ALLOWED_NEGATIVE_DEVIATION = 0.3 # How many m/s below target VSI before heavy penalty kicks in (e.g. target 1, penalty starts below -1)

# --- Pitch/Altitude Control Method ---
# Options: 'AP_PITCH', 'SAS_STABILITY'
# 'SAS_STABILITY' often works better if AP struggles with pitch target
PITCH_CONTROL_METHOD = 'AP_PITCH'
TARGET_INITIAL_PITCH = 15.0 # Degrees (Adjust based on aircraft and desired climb)

# --- PID Controller Class (Unchanged) ---
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-1, 1), integral_limits=(-50, 50)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits # Prevent integral windup

        self._integral = 0
        self._last_error = 0
        self._last_time = time.time()
        self.output = 0 # Store last output

    def update(self, measured_value):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0:
            return self.output

        # Calculate heading error (handle 360/0 degree wrap-around)
        error = self.setpoint - measured_value
        while error > 180: error -= 360
        while error <= -180: error += 360

        p_term = self.Kp * error

        self._integral += error * dt
        self._integral = max(self.integral_limits[0], min(self.integral_limits[1], self._integral))
        i_term = self.Ki * self._integral

        derivative = (error - self._last_error) / dt
        d_term = self.Kd * derivative

        output = p_term + i_term + d_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        self._last_error = error
        self._last_time = current_time
        self.output = output
        return output

    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._last_time = time.time()
        self.output = 0

