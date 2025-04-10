# Required Libraries: pip install krpc pygad numpy matplotlib
import krpc
import time
import math
import pygad
import numpy as np
import matplotlib.pyplot as plt # Needed for ga_instance.plot_fitness()

# --- Configuration ---
KSP_CONNECTION_NAME = 'GA_PID_Tuner_Reload_SAS_Robust' # Updated name
QUICKSAVE_NAME = "SAVEGAME" # <<< NAME OF YOUR PRE-MADE QUICKSAVE FILE >>>

TARGET_HEADING = 45      # Target heading in degrees
TARGET_VERTICAL_SPEED = 1.0 # Target vertical speed in m/s
SIMULATION_DURATION = 15 # How many seconds to test each set of PID gains
CONTROL_UPDATE_INTERVAL = 0.05 # How often to update PID and controls
INITIAL_THROTTLE = 1.0   # Initial throttle setting for the test flight

# --- Fitness Function Weights ---
HEADING_WEIGHT = 1.0
VSI_WEIGHT = 0.5
DIVING_PENALTY_MULTIPLIER = 5.0
VSI_ALLOWED_NEGATIVE_DEVIATION = 2.0

# --- Pitch/Altitude Control Method ---
PITCH_CONTROL_METHOD = 'AP_PITCH'
TARGET_INITIAL_PITCH = 15.0

# --- PID Controller Class (Unchanged) ---
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-1, 1), integral_limits=(-50, 50)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self._integral = 0
        self._last_error = 0
        self._last_time = time.time()
        self.output = 0

    def update(self, measured_value):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt <= 0: return self.output
        error = self.setpoint - measured_value
        while error > 180: error -= 360
        while error <= -180: error += 360
        p_term = self.Kp * error
        self._integral += error * dt
        self._integral = max(self.integral_limits[0], min(self.integral_limits[1], self._integral))
        i_term = self.Ki * self._integral
        if dt <= 0: derivative = 0
        else: derivative = (error - self._last_error) / dt
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

# --- Callback Function for Inter-Generation Reload & SAS (MODIFIED FOR ROBUSTNESS) ---
def generation_reload_callback(ga_instance):
    """
    Called by PyGAD after each generation. Prints status, attempts to reload
    a quicksave, waits robustly, enables SAS, and announces the start.
    """
    completed_gen = ga_instance.generations_completed
    next_gen_num = completed_gen + 1

    print(f"\n G----------------------------------------------------------G")
    print(f"| Generation {completed_gen} Finished.")
    try:
        # Display best fitness (unchanged)
        last_gen_fitness = ga_instance.last_generation_fitness
        if last_gen_fitness is not None and len(last_gen_fitness) > 0:
             best_fitness_this_gen = np.max(last_gen_fitness)
             print(f"| Best Fitness in Generation {completed_gen}: {best_fitness_this_gen:.4f}")
        else: print("| (Could not retrieve last generation fitness)")
    except Exception as e: print(f"| (Error retrieving fitness: {e})")

    if completed_gen < ga_instance.num_generations:
        print(f"| Preparing for Generation {next_gen_num}: Reloading & Enabling SAS")
        print(f" G----------------------------------------------------------G")

        reload_success = False
        sas_enabled = False
        temp_conn = None
        vessel = None # Initialize vessel to None

        try:
            # --- Step 1: Reload ---
            print("    Connecting to kRPC for reload...")
            temp_conn = krpc.connect(name=f'{KSP_CONNECTION_NAME}_Reloader_{next_gen_num}')
            sc = temp_conn.space_center
            print(f"    Issuing quickload command for '{QUICKSAVE_NAME}'...")
            sc.quickload(QUICKSAVE_NAME)

            # --- Step 2: Robust Wait & Vessel Check ---
            # Initial wait - adjust based on typical load time
            wait_duration = 20.0 # <<< INCREASED & TUNE THIS! Try 20, 25, 30+ if needed >>>
            print(f"    Quickload initiated. Initial wait: {wait_duration:.1f} seconds...")
            for i in range(int(wait_duration), 0, -1):
                 print(f"      Load time remaining: {i:2d}s... \r", end='', flush=True)
                 time.sleep(1)
            print("                                          \r", end='')
            print("    Initial load wait finished.")

            # Active check for vessel readiness
            vessel_ready_wait_max = 15 # Max extra seconds to wait for vessel object
            print(f"    Actively checking for vessel readiness (max {vessel_ready_wait_max}s)...")
            vessel_found = False
            for i in range(vessel_ready_wait_max * 2): # Check twice per second
                try:
                    vessel = sc.active_vessel
                    # Try accessing a property that requires the vessel to be loaded
                    vessel_name = vessel.name
                    if vessel is not None and vessel_name:
                         print(f"    Active vessel '{vessel_name}' found and responsive.")
                         vessel_found = True
                         reload_success = True # Mark reload as successful *only* if vessel is found
                         break # Exit loop if vessel found
                except (krpc.error.RPCError, AttributeError, TypeError):
                    # Ignore errors while waiting for vessel to become available
                    # TypeError can happen if vessel is None temporarily
                    print(f"      Waiting for vessel... ({i/2.0 + 0.5:.1f}s)\r", end='', flush=True)
                    pass
                time.sleep(0.5)

            if not vessel_found:
                print("\n!! ERROR: Failed to get responsive active vessel after quickload and extended wait.")
                raise RuntimeError("Failed to get active vessel after quickload.") # Force failure

            # --- Step 3: Enable SAS (only if reload succeeded) ---
            if reload_success:
                print("    Attempting to enable SAS...")
                control = vessel.control
                ap = vessel.auto_pilot

                if control.sas_available:
                    control.sas = True
                    time.sleep(0.3) # Slightly longer pause for SAS
                    if control.sas:
                        print("    SAS Enabled successfully.")
                        sas_enabled = True
                        try:
                            if hasattr(sc, 'SASMode'):
                                ap.sas_mode = sc.SASMode.stability_assist
                                print("    SAS Mode set to Stability Assist.")
                            else: print("    (SASMode attribute not found, basic SAS enabled)")
                        except krpc.error.RPCError as sas_mode_e: print(f"    Warning: Could not set SAS mode: {sas_mode_e}")
                    else: print("!! WARNING: Attempted to enable SAS, but it did not engage.")
                else: print("!! WARNING: SAS is not available on this vessel.")

        except ConnectionRefusedError: print("!! ERROR: KSP Connection Refused during reload/SAS attempt."); reload_success = False
        except krpc.error.RPCError as e: print(f"!! ERROR: kRPC Error during reload/SAS: {e}"); reload_success = False
        except RuntimeError as e: print(f"!! ERROR: {e}"); reload_success = False # Catch our explicit failure
        except Exception as e: print(f"!! ERROR: Unexpected Error during reload/SAS: {type(e).__name__}: {e}"); reload_success = False
        finally:
            if temp_conn:
                try: temp_conn.close(); print("    Reload/SAS connection closed.")
                except Exception: pass

        # --- Step 4: Check Overall Success ---
        if not reload_success:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! RELOAD/SETUP FAILED. Stopping script.               !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise RuntimeError(f"Failed during reload/SAS setup for Generation {next_gen_num}.")
        else:
            print(f"\n>>> Now training Generation {next_gen_num}... <<<")
            time.sleep(1)

    else:
         print(f"\n G----------------------------------------------------------G")
         print("| Final generation complete. No reload/SAS needed.")


# --- Fitness Function (Added Debug Prints) ---
def fitness_func(ga_instance, solution, solution_idx):
    """
    Evaluates a set of PID gains (solution) by running a KSP simulation segment.
    Fitness is based on minimizing heading error AND minimizing deviation
    from the TARGET_VERTICAL_SPEED, penalizing diving. Higher fitness is better.
    """
    Kp, Ki, Kd = solution
    gen_num = ga_instance.generations_completed + 1
    print(f"\n--- Evaluating Solution {solution_idx} (Gen {gen_num}): Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} ---")

    conn = None
    total_abs_heading_error = 0
    total_vertical_speed = 0
    num_updates = 0
    fitness = 0.0

    try:
        # Connect to KSP
        print("  Connecting to kRPC...")
        conn = krpc.connect(name=f'{KSP_CONNECTION_NAME}_Eval_{solution_idx}')
        print("  Getting SpaceCenter objects...")
        vessel = conn.space_center.active_vessel
        # <<< DEBUG PRINT >>>
        if vessel is None:
             print("!! CRITICAL ERROR: vessel object is None immediately after connect!")
             return 0.0 # Fail fast
        print(f"  Vessel found: {vessel.name}")
        control = vessel.control
        ap = vessel.auto_pilot
        surface_ref = vessel.surface_reference_frame
        flight = vessel.flight(surface_ref)
        print("  SpaceCenter objects obtained.")

        # --- Simulation Setup ---
        print(f"  Setting up simulation: Target Head={TARGET_HEADING}, Target VSI≈{TARGET_VERTICAL_SPEED}, Dur={SIMULATION_DURATION}s")
        pid_controller = PID(Kp, Ki, Kd, TARGET_HEADING)
        pid_controller.reset()

        # Prepare vessel state
        print(f"  Setting initial throttle: {INITIAL_THROTTLE}")
        control.throttle = INITIAL_THROTTLE
        # Ensure clean state before applying eval method
        print("  Ensuring SAS/AP are disabled for evaluation start...")
        control.sas = False
        if ap.engaged: ap.disengage()
        time.sleep(0.2) # Pause for commands
        print(f"  SAS status before eval method: {control.sas}, AP status: {ap.engaged}")

        # --- Apply Pitch/Roll Control Method ---
        # (Error handling within these blocks might be needed if they fail often)
        if PITCH_CONTROL_METHOD == 'AP_PITCH':
            print(f"  Engaging Autopilot: Target Pitch={TARGET_INITIAL_PITCH} deg, Target Roll=0 deg")
            ap.reference_frame = surface_ref
            ap.target_pitch = TARGET_INITIAL_PITCH
            ap.target_roll = 0
            if not ap.engaged: ap.engage()
            stabilization_wait = 2.0
            print(f"  Waiting {stabilization_wait}s for initial AP set...")
            time.sleep(stabilization_wait)
            print(f"  AP status after wait: Engaged={ap.engaged}, TargetPitch={ap.target_pitch:.1f}")
        elif PITCH_CONTROL_METHOD == 'SAS_STABILITY':
             if vessel.control.sas_available:
                 print("  Engaging SAS (Stability Assist) for evaluation...")
                 control.sas = True; time.sleep(0.1)
                 try:
                      if hasattr(conn.space_center, 'SASMode'):
                          ap.sas_mode = conn.space_center.SASMode.stability_assist
                          print("  SAS Mode set to Stability Assist.")
                      else: print("  (SASMode attribute not found, basic SAS enable)")
                      stabilization_wait = 2.0
                      print(f"  Waiting {stabilization_wait}s for initial SAS set...")
                      time.sleep(stabilization_wait)
                      print(f"  SAS status after wait: {control.sas}")
                 except krpc.error.RPCError as e:
                      print(f"  Warning: Could not set SAS mode: {e}. Relying on basic SAS enable.")
                      control.sas = True; time.sleep(1)
             else:
                 print("!! Warning: SAS unavailable for SAS_STABILITY method! Aborting solution.")
                 return 0.0
        else:
             print(f"!! Error: Unknown PITCH_CONTROL_METHOD '{PITCH_CONTROL_METHOD}'. Aborting solution.")
             return 0.0

        print("  Proceeding to simulation loop.")

        # --- Simulation Loop ---
        print("  Starting PID control loop...")
        start_time = time.time()
        last_print_time = start_time
        loop_error = None # Variable to store loop error if any

        try:
            while time.time() - start_time < SIMULATION_DURATION:
                # <<< DEBUG: Check connection inside loop periodically? (Optional, adds overhead) >>>
                # if num_updates % 50 == 0: conn.krpc.get_status() # Throws error if disconnected

                current_heading = flight.heading
                current_vsi = flight.vertical_speed
                num_updates += 1

                heading_error = TARGET_HEADING - current_heading
                while heading_error > 180: heading_error -= 360
                while heading_error <= -180: heading_error += 360
                total_abs_heading_error += abs(heading_error)

                yaw_input = pid_controller.update(current_heading)
                control.yaw = yaw_input # PID controls yaw

                total_vertical_speed += current_vsi

                if time.time() - last_print_time > 3.0:
                     print(f"    t={time.time()-start_time:.1f}s, Head={current_heading:.1f}, Err={heading_error:.1f}, Yaw={yaw_input:.2f}, VSI={current_vsi:.4f}")
                     last_print_time = time.time()

                time.sleep(CONTROL_UPDATE_INTERVAL)
        except krpc.error.RPCError as loop_e:
             print(f"!! kRPC Error INSIDE simulation loop: {loop_e}")
             loop_error = loop_e # Store error to potentially reduce fitness later
        except Exception as loop_e:
             print(f"!! Unexpected Error INSIDE simulation loop: {type(loop_e).__name__}: {loop_e}")
             loop_error = loop_e

        # --- Simulation End & Cleanup ---
        print("  Simulation loop finished.")
        # Reset controls immediately after loop
        try:
            control.yaw = 0
            # Maybe reduce throttle slightly?
            # control.throttle = 0.8
        except krpc.error.RPCError:
            print("  kRPC error resetting yaw after loop (likely disconnected).")


        # --- Calculate Fitness ---
        if num_updates > 0 and loop_error is None: # Only calculate fitness if loop ran and had no errors
            average_abs_heading_error = total_abs_heading_error / num_updates
            average_vertical_speed = total_vertical_speed / num_updates
            vsi_deviation = abs(average_vertical_speed - TARGET_VERTICAL_SPEED)

            if average_vertical_speed < (TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION):
                print(f"    Applying diving penalty (Avg VSI {average_vertical_speed:.2f} < {TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION:.2f})")
                vsi_component_penalty = vsi_deviation * DIVING_PENALTY_MULTIPLIER
            else:
                vsi_component_penalty = vsi_deviation

            combined_error = (HEADING_WEIGHT * average_abs_heading_error) + \
                             (VSI_WEIGHT * vsi_component_penalty)
            fitness = 1.0 / (combined_error + 1e-6)

            print(f"  Avg Abs Heading Error: {average_abs_heading_error:.4f}")
            print(f"  Average Vertical Speed: {average_vertical_speed:.4f} (Target: {TARGET_VERTICAL_SPEED})")
            print(f"  VSI Deviation (Penalized): {vsi_component_penalty:.4f}")
            print(f"  Combined Weighted Error: {combined_error:.4f} -> Fitness: {fitness:.4f}")

        elif loop_error is not None:
             print(f"  Simulation loop interrupted by error. Assigning low fitness.")
             fitness = 0.0 # Assign low fitness if loop failed
        else: # num_updates == 0
            print("  No updates occurred during simulation loop, fitness set to 0.")
            fitness = 0.0

    except ConnectionRefusedError: print("!! KSP Connection Refused."); fitness = 0.0
    except krpc.error.RPCError as e: print(f"!! kRPC Error during setup/eval: {e}"); fitness = 0.0
    except Exception as e:
        print(f"!! Unexpected Error during setup/eval: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc(); fitness = 0.0
    finally:
        # Cleanup evaluation-specific controls (AP/SAS used during eval)
        if conn: # Only cleanup if connection was potentially established
            try:
                print("  Cleaning up controls after evaluation...")
                # Check if control object exists before using it
                if 'control' in locals() and control is not None:
                    control.yaw = 0 # Ensure yaw is zeroed
                    if PITCH_CONTROL_METHOD == 'SAS_STABILITY':
                        # Only turn off SAS if it was the method used *during* this eval
                        control.sas = False
                # Check if ap object exists before using it
                if 'ap' in locals() and ap is not None and PITCH_CONTROL_METHOD == 'AP_PITCH':
                     if ap.engaged: ap.disengage()
            except krpc.error.RPCError: print("  kRPC error during final cleanup (connection likely lost).")
            except Exception as e_clean: print(f"  Unexpected error during final cleanup: {e_clean}")
            finally:
                 try: conn.close(); print("  Closing kRPC connection for evaluation.")
                 except Exception: pass # Ignore errors closing already closed/failed connection

    # <<< DEBUG PRINT >>>
    print(f"  Fitness returned for Sol {solution_idx} (Gen {gen_num}): {fitness:.4f}")
    return fitness

# --- Genetic Algorithm Setup ---
# (Gene Space and GA Parameters remain the same)
gene_space = [
    {'low': 0.0, 'high': 2.5}, {'low': 0.0, 'high': 2.5}, {'low': 0.0, 'high': 2.5}
]
num_genes = len(gene_space)
num_generations = 6 # User specified lower value
num_parents_mating = 2
sol_per_pop = 6
parent_selection_type = "sss"
keep_parents = 2
mutation_type = "random"
mutation_percent_genes = 30

print("\n=== Initializing Genetic Algorithm ===")
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       allow_duplicate_genes=False,
                       on_generation=generation_reload_callback # Uses the modified callback
                       )

# --- Run the GA ---
print("\n=== Starting Genetic Algorithm Run ===")
print(f"Target Heading: {TARGET_HEADING} degrees")
print(f"Target Vertical Speed: {TARGET_VERTICAL_SPEED} m/s")
print(f"Simulation Duration per Solution: {SIMULATION_DURATION} seconds")
print(f"Pitch Control Method during Eval: {PITCH_CONTROL_METHOD}")
if PITCH_CONTROL_METHOD == 'AP_PITCH': print(f"Target Initial Pitch during Eval: {TARGET_INITIAL_PITCH} degrees")
print(f"Quicksave Name for Reload: '{QUICKSAVE_NAME}'")
print(f"Initial Load Wait Time: {wait_duration}s (Tune in callback if needed!)") # Referencing wait_duration from callback scope (might be better to define globally)
print("\n!!! IMPORTANT !!!")
print(f"Ensure quicksave '{QUICKSAVE_NAME}' exists and loads correctly manually.")
print("The script will reload and enable SAS between generations.")
input("Press Enter to start the GA optimization...")

try:
    ga_instance.run()
except KeyboardInterrupt: print("\n!!! GA Run Interrupted by User !!!")
except RuntimeError as e: print(f"\n!!! GA Run Stopped due to Error: {e} !!!")
except Exception as e: # Catch any other unexpected errors during run
     print(f"\n!!! UNEXPECTED ERROR during GA run: {type(e).__name__}: {e} !!!")
     import traceback; traceback.print_exc()


# --- Results ---
print("\n=== Genetic Algorithm Finished ===")
# (Plotting and results section uses the fixed version from previous step)
if ga_instance.generations_completed > 0:
    try:
        print("  Generating fitness plot...")
        ga_instance.plot_fitness()
    except ImportError: print("\nInstall matplotlib to see the fitness plot.")
    except AttributeError: print("\nWarning: PyGAD version might not support plot_fitness().")
    except Exception as e: print(f"\nCould not display plot: {type(e).__name__}: {e}")
    try:
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("\nBest solution found:")
        if hasattr(ga_instance, 'best_solution_generation'): print(f"  Generation: {ga_instance.best_solution_generation}")
        print(f"  Index in Last Pop: {solution_idx}")
        print(f"  Gains (Kp, Ki, Kd): ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
        print(f"  Fitness value = {solution_fitness:.6f}")
    except Exception as e: print(f"\nError retrieving best solution details: {e}")
else: print("\nNo generations were completed.")
print("\nOptimization complete.")