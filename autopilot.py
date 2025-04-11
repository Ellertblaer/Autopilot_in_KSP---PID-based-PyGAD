# Required Libraries: pip install krpc numpy
import krpc
import time
import math
import numpy as np # Still useful for clamping, etc.

# --- User Configuration ---

# !! IMPORTANT: REPLACE THESE WITH THE BEST VALUES FOUND BY YOUR GA !!
BEST_KP = 0.773593  # Replace with your best Kp
BEST_KI = 0.756765  # Replace with your best Ki
BEST_KD = 2.159699  # Replace with your best Kd

# Flight Targets
TARGET_HEADING = 180      # Target heading in degrees
TARGET_VERTICAL_SPEED = 1.0 # Target vertical speed in m/s

# VSI Control (Proportional) - TUNE THESE!
VSI_P_GAIN = 0.75          # How aggressively pitch adjusts for VSI error (Try 0.1 to 1.0)
BASE_PITCH = 1.0          # Base autopilot pitch target (degrees). Adjust for your aircraft's level/climb performance.
MIN_PITCH = 0.0         # Minimum allowed autopilot pitch target
MAX_PITCH = 2.0          # Maximum allowed autopilot pitch target

# Other Settings
KSP_CONNECTION_NAME = 'Continuous_PID_Flight'
CONTROL_UPDATE_INTERVAL = 0.05 # How often to update PID and controls (seconds)
INITIAL_THROTTLE = 1.0 # Initial throttle (can be adjusted manually in KSP later)

# --- PID Controller Class (Copied from your GA script) ---
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
            return self.output # Avoid division by zero or negative dt

        # Calculate heading error (handle 360/0 degree wrap-around)
        error = self.setpoint - measured_value
        while error > 180: error -= 360
        while error <= -180: error += 360

        p_term = self.Kp * error

        self._integral += error * dt
        self._integral = np.clip(self._integral, self.integral_limits[0], self.integral_limits[1]) # Use numpy clip
        i_term = self.Ki * self._integral

        # Avoid division by zero if dt is extremely small
        if dt < 1e-6:
             derivative = 0
        else:
             derivative = (error - self._last_error) / dt
        d_term = self.Kd * derivative

        output = p_term + i_term + d_term
        output = np.clip(output, self.output_limits[0], self.output_limits[1]) # Use numpy clip

        self._last_error = error
        self._last_time = current_time
        self.output = output
        return output

    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._last_time = time.time()
        self.output = 0

# --- Main Execution ---
if __name__ == "__main__":
    conn = None
    try:
        print(f"Connecting to kRPC as '{KSP_CONNECTION_NAME}'...")
        conn = krpc.connect(name=KSP_CONNECTION_NAME)
        vessel = conn.space_center.active_vessel
        control = vessel.control
        ap = vessel.auto_pilot
        surface_ref = vessel.surface_reference_frame
        flight = vessel.flight(surface_ref) # Use surface flight info

        print("Connection successful.")
        print(f"Target Heading: {TARGET_HEADING} deg")
        print(f"Target Vertical Speed: {TARGET_VERTICAL_SPEED} m/s")
        print(f"Using PID Gains: Kp={BEST_KP:.4f}, Ki={BEST_KI:.4f}, Kd={BEST_KD:.4f}")
        print(f"VSI P-Gain: {VSI_P_GAIN}, Base Pitch: {BASE_PITCH} deg")

        # --- Setup ---
        print("Setting up flight controls...")
        heading_pid = PID(BEST_KP, BEST_KI, BEST_KD, TARGET_HEADING)
        heading_pid.reset()

        # Ensure autopilot is set up for pitch/roll/yaw control relative to surface
        ap.reference_frame = surface_ref
        ap.target_roll = 0  # Keep wings level
        ap.target_pitch = BASE_PITCH # Set initial pitch
        ap.engage() # Engage autopilot for pitch and roll

        control.throttle = INITIAL_THROTTLE # Set initial throttle

        # Disengage SAS if it was on, AP now handles stability
        control.sas = False
        time.sleep(1) # Give AP time to stabilize initially

        print("--- Starting Continuous Control Loop (Press Ctrl+C to Stop) ---")

        # --- Main Control Loop ---
        while True:
            try:
                # Get current state
                current_heading = flight.heading
                current_vsi = flight.vertical_speed

                # --- Heading Control (PID) ---
                yaw_input = heading_pid.update(current_heading)
                control.yaw = yaw_input # Apply yaw control directly

                # --- Vertical Speed Control (Proportional Pitch Adjustment) ---
                vsi_error = TARGET_VERTICAL_SPEED - current_vsi
                pitch_adjustment = VSI_P_GAIN * vsi_error
                new_target_pitch = BASE_PITCH + pitch_adjustment

                # Clamp the target pitch to reasonable limits
                new_target_pitch = np.clip(new_target_pitch, MIN_PITCH, MAX_PITCH)

                # Update the autopilot's pitch target
                ap.target_pitch = new_target_pitch

                # --- Print Status Periodically ---
                print(f"Head: {current_heading:6.1f}/{TARGET_HEADING:.0f} | "
                      f"VSI: {current_vsi:6.1f}/{TARGET_VERTICAL_SPEED:.1f} | "
                      f"Yaw: {yaw_input:+.2f} | "
                      f"Pitch Cmd: {new_target_pitch:5.1f} | "
                      f"VSI Err: {vsi_error:+.1f} \r", end="")

                # Wait for the next control cycle
                time.sleep(CONTROL_UPDATE_INTERVAL)

            except krpc.error.RPCError as e:
                print(f"\n!!! kRPC Error during loop: {e} - Check vessel state (crashed?). Exiting.")
                break
            except krpc.error.ConnectionError:
                print("\n!!! kRPC Connection Lost. Exiting.")
                break

    except krpc.error.ConnectionError as e:
        print(f"Error: Cannot connect to kRPC server: {e}")
        print("Is KSP running with the kRPC mod installed and server started?")
    except KeyboardInterrupt:
        print("\n--- Control loop interrupted by user (Ctrl+C) ---")
    except Exception as e:
        print(f"\n!!! An unexpected error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Cleanup ---
        if conn:
            print("\nCleaning up controls and closing connection...")
            try:
                # Check if objects exist before trying to use them
                if 'ap' in locals() and ap and ap.engaged:
                    print("  Disengaging autopilot.")
                    ap.disengage()
                if 'control' in locals() and control:
                    print("  Centering controls.")
                    control.yaw = 0
                    control.pitch = 0
                    control.roll = 0
                    # Optionally reduce throttle on exit
                    # control.throttle = 0.5
            except krpc.error.RPCError:
                print("  kRPC Error during cleanup (connection likely lost).")
            except Exception as e_clean:
                print(f"  Unexpected error during cleanup: {e_clean}")
            finally:
                conn.close()
                print("Connection closed.")
        else:
             print("No active connection to close.")

    print("Script finished.")