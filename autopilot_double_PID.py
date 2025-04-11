# Required Libraries: pip install krpc numpy
import krpc
import time
import math
import numpy as np # Used for clamping

# --- User Configuration ---

# !! HEADING PID: REPLACE THESE WITH THE BEST VALUES FOUND BY YOUR GA !!
BEST_KP = 1.0  # Replace with your best Heading Kp
BEST_KI = 0.5  # Replace with your best Heading Ki
BEST_KD = 0.2  # Replace with your best Heading Kd

# !! VSI PID: TUNE THESE VALUES CAREFULLY !!
# Start with small values and increase gradually. Tuning VSI PID can be tricky.
VSI_KP = 0.3     # Proportional gain for VSI error -> pitch adjustment
VSI_KI = 0.05    # Integral gain for VSI error -> pitch adjustment
VSI_KD = 0.1     # Derivative gain for VSI error -> pitch adjustment

# Flight Targets
TARGET_HEADING = 180      # Target heading in degrees
TARGET_VERTICAL_SPEED = 1.0 # Target vertical speed in m/s

# Pitch Control Settings
BASE_PITCH = 5.0          # Base autopilot pitch target (degrees). The VSI PID output ADDS to this.
                          # Adjust for your aircraft's typical level/climb pitch.
MIN_PITCH = -5.0         # Minimum allowed final autopilot pitch target
MAX_PITCH = 25.0          # Maximum allowed final autopilot pitch target

# VSI PID Output Limits (limits the *adjustment* the PID adds to BASE_PITCH)
VSI_PID_OUTPUT_LIMITS = (-10.0, 15.0) # e.g., Max decrease of 10deg, max increase of 15deg from BASE_PITCH
VSI_PID_INTEGRAL_LIMITS = (-100.0, 100.0) # Prevents VSI integral windup, adjust if needed

# Other Settings
KSP_CONNECTION_NAME = 'Continuous_Dual_PID_Flight'
CONTROL_UPDATE_INTERVAL = 0.05 # How often to update PIDs and controls (seconds)
INITIAL_THROTTLE = 1.0 # Initial throttle (can be adjusted manually in KSP later)

# --- PID Controller Class (Unchanged - Copied from your GA script) ---
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

        # Calculate error (Specific logic for heading wrap-around is handled outside if needed)
        error = self.setpoint - measured_value

        # --- Heading Specific Wrap-Around (Only for Heading PID) ---
        # We need context to know if this PID is for heading. Let's assume
        # the calling code handles wrap-around *before* passing measured_value
        # OR we add a flag/check. Simpler to handle it outside for now.
        # ---> Let's modify this: The PID *using* this class will handle wrap-around logic.
        # ---> Update: The original PID class *did* handle wrap-around. Reinstating that logic.

        # Calculate heading error (handle 360/0 degree wrap-around) if setpoint is potentially angular
        # A simple heuristic: if setpoint is >= 0 and <= 360, assume it *could* be heading
        # This isn't perfect, but avoids adding flags to the class for now.
        # A better way would be a subclass or a flag during init.
        # Reverting to original wrap-around logic inside PID as it was in GA script:
        is_heading_like = True # Let's assume PID might be used for heading
        if is_heading_like:
            temp_error = self.setpoint - measured_value # Calculate error normally first
            while temp_error > 180: temp_error -= 360
            while temp_error <= -180: temp_error += 360
            error = temp_error # Use the wrapped-around error if applicable

        p_term = self.Kp * error

        self._integral += error * dt
        self._integral = np.clip(self._integral, self.integral_limits[0], self.integral_limits[1]) # Use numpy clip
        i_term = self.Ki * self._integral

        # Avoid division by zero if dt is extremely small
        if dt < 1e-6:
             derivative = 0
        else:
             # Prevent division by zero on the very first update after reset
             if self._last_time == 0: # Check if it's the first run
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
        # Setting last_time to 0 might be better to indicate it hasn't run yet
        # self._last_time = time.time()
        self._last_time = 0 # Indicate first run for derivative calculation
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
        print(f"Heading PID Gains (Kp, Ki, Kd): ({BEST_KP:.4f}, {BEST_KI:.4f}, {BEST_KD:.4f})")
        print(f"VSI PID Gains (Kp, Ki, Kd):     ({VSI_KP:.4f}, {VSI_KI:.4f}, {VSI_KD:.4f})")
        print(f"Base Pitch: {BASE_PITCH} deg, Final Pitch Limits: [{MIN_PITCH}, {MAX_PITCH}] deg")
        print(f"VSI PID Output Adj Limits: [{VSI_PID_OUTPUT_LIMITS[0]}, {VSI_PID_OUTPUT_LIMITS[1]}] deg")


        # --- Setup PID Controllers ---
        print("Setting up PID controllers...")
        # Heading PID
        heading_pid = PID(BEST_KP, BEST_KI, BEST_KD, TARGET_HEADING,
                          output_limits=(-1, 1)) # Yaw input is -1 to 1
        heading_pid.reset()

        # Vertical Speed PID
        vsi_pid = PID(VSI_KP, VSI_KI, VSI_KD, TARGET_VERTICAL_SPEED,
                      output_limits=VSI_PID_OUTPUT_LIMITS,
                      integral_limits=VSI_PID_INTEGRAL_LIMITS)
        vsi_pid.reset()


        # --- Setup Autopilot and Controls ---
        print("Setting up flight controls...")
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

                # --- Heading Control (PID -> Yaw) ---
                # PID class now handles wrap-around internally
                yaw_input = heading_pid.update(current_heading)
                control.yaw = yaw_input # Apply yaw control directly

                # --- Vertical Speed Control (PID -> Pitch Adjustment) ---
                # Update VSI PID controller
                pitch_adjustment = vsi_pid.update(current_vsi)

                # Calculate the new target pitch by adding the PID adjustment to the base pitch
                new_target_pitch = BASE_PITCH + pitch_adjustment

                # Clamp the final target pitch to the overall aircraft limits
                new_target_pitch = np.clip(new_target_pitch, MIN_PITCH, MAX_PITCH)

                # Update the autopilot's pitch target
                ap.target_pitch = new_target_pitch

                # --- Print Status Periodically ---
                # Calculate heading error for printing (with wrap-around)
                h_err = TARGET_HEADING - current_heading
                while h_err > 180: h_err -= 360
                while h_err <= -180: h_err += 360
                v_err = TARGET_VERTICAL_SPEED - current_vsi

                print(f"Head: {current_heading:6.1f}/{TARGET_HEADING:.0f} (Err:{h_err: 5.1f}) | "
                      f"VSI: {current_vsi:6.1f}/{TARGET_VERTICAL_SPEED:.1f} (Err:{v_err: 5.1f}) | "
                      f"Yaw: {yaw_input:+.2f} | "
                      f"PitchCmd: {new_target_pitch:5.1f} (Adj:{pitch_adjustment: 5.1f}) \r", end="")

                # Wait for the next control cycle
                time.sleep(CONTROL_UPDATE_INTERVAL)

            except krpc.error.RPCError as e:
                # Handle potential disconnects or vessel destruction gracefully
                print(f"\n!!! kRPC Error during loop: {e}")
                print("!!! Check vessel state (destroyed?) or kRPC server. Exiting.")
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
                # Check if objects exist and are valid before trying to use them
                # Check connection state before RPC calls during cleanup
                if conn.krpc.get_status().version: # Simple check if connection seems alive
                    if 'ap' in locals() and ap:
                         # Check if vessel still exists before accessing autopilot state
                         try:
                              vessel_exists_check = vessel.name # Simple check
                              if ap.engaged:
                                   print("  Disengaging autopilot.")
                                   ap.disengage()
                         except krpc.error.RPCError:
                              print("  Vessel likely destroyed, cannot disengage AP.")
                    if 'control' in locals() and control:
                         try:
                             print("  Centering controls.")
                             control.yaw = 0
                             control.pitch = 0
                             control.roll = 0
                             # Optionally reduce throttle on exit
                             # control.throttle = 0.5
                         except krpc.error.RPCError:
                             print("  Vessel likely destroyed, cannot reset controls.")
                else:
                    print("  Connection seems closed, skipping RPC cleanup.")

            except krpc.error.RPCError as e_rpc_clean:
                 # Catch errors specifically during cleanup RPC calls
                 print(f"  kRPC Error during cleanup (connection likely lost or vessel gone): {e_rpc_clean}")
            except Exception as e_clean:
                 print(f"  Unexpected error during cleanup: {e_clean}")
            finally:
                 conn.close()
                 print("Connection closed.")
        else:
             print("No active connection to close.")

    print("Script finished.")