# Required Libraries: pip install krpc pygad numpy matplotlib
import krpc
import time
import math
import pygad
import numpy as np
import matplotlib.pyplot as plt # Needed for ga_instance.plot_fitness()

# --- Configuration ---
KSP_CONNECTION_NAME = 'GA_PID_Tuner_Reload_FIXED' # Updated name
QUICKSAVE_NAME = "GAD_Start_State" # <<< NAME OF YOUR PRE-MADE QUICKSAVE FILE >>>

TARGET_HEADING = 45      # Target heading in degrees
TARGET_VERTICAL_SPEED = 1.0 # Target vertical speed in m/s
SIMULATION_DURATION = 15 # How many seconds to test each set of PID gains
CONTROL_UPDATE_INTERVAL = 0.05 # How often to update PID and controls
INITIAL_THROTTLE = 1.0   # Initial throttle setting for the test flight

# --- Fitness Function Weights ---
HEADING_WEIGHT = 2.5
VSI_WEIGHT = 0.5
DIVING_PENALTY_MULTIPLIER = 3.0
VSI_ALLOWED_NEGATIVE_DEVIATION = 1.5

# --- Pitch/Altitude Control Method ---
PITCH_CONTROL_METHOD = 'AP_PITCH'
TARGET_INITIAL_PITCH = 15.0

# GA Parameters
num_generations = 3
num_parents_mating = 2
sol_per_pop = 6
parent_selection_type = "sss"
keep_parents = 2
mutation_type = "random"
mutation_percent_genes = 30

# --- PID Controller Class (Unchanged) ---
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(-1, 1), integral_limits=(-50, 50)):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits; self.integral_limits = integral_limits
        self._integral = 0; self._last_error = 0
        self._last_time = time.time(); self.output = 0

    def update(self, measured_value):
        current_time = time.time(); dt = current_time - self._last_time
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
        self._last_error = error; self._last_time = current_time
        self.output = output; return output

    def reset(self):
        self._integral = 0; self._last_error = 0
        self._last_time = time.time(); self.output = 0

# --- Callback Function for Inter-Generation Reload (Corrected) ---
def generation_reload_callback(ga_instance):
    """
    Called by PyGAD after each generation. Prints status, attempts to reload
    a quicksave, waits robustly, and announces the start of the next generation.
    FIXED: Removed misplaced SAS line. Added robust wait after load.
    """
    completed_gen = ga_instance.generations_completed
    next_gen_num = completed_gen + 1

    # <<< REMOVED Misplaced SAS line here >>>

    print(f"\n G----------------------------------------------------------G")
    print(f"| Generation {completed_gen} Finished.")
    try:
        last_gen_fitness = ga_instance.last_generation_fitness
        if last_gen_fitness is not None and len(last_gen_fitness) > 0:
             best_fitness_this_gen = np.max(last_gen_fitness)
             print(f"| Best Fitness in Generation {completed_gen}: {best_fitness_this_gen:.4f}")
        else: print("| (Could not retrieve last generation fitness)")
    except Exception as e: print(f"| (Error retrieving fitness: {e})")

    if completed_gen < ga_instance.num_generations:
        print(f"| Attempting to reload quicksave: '{QUICKSAVE_NAME}'")
        print(f" G----------------------------------------------------------G")
        reload_success = False; temp_conn = None
        try:
            print("    Connecting to kRPC for reload...")
            temp_conn = krpc.connect(name=f'{KSP_CONNECTION_NAME}_Reloader_{next_gen_num}')
            sc = temp_conn.space_center
            print(f"    Issuing quickload command for '{QUICKSAVE_NAME}'...")
            sc.quickload(QUICKSAVE_NAME)
            wait_duration = 15.0 # <<< TUNE THIS WAIT TIME! 15 might be too short >>>
            print(f"    Quickload initiated. Waiting {wait_duration:.1f} seconds for KSP to load...")
            for i in range(int(wait_duration), 0, -1):
                 print(f"      Load time remaining: {i:2d}s... \r", end='', flush=True); time.sleep(1)
            print("                                          \r", end=''); print("    Load wait finished.")

            # Add active check for vessel readiness after load
            vessel_ready_wait_max = 15; vessel_found = False
            print(f"    Actively checking for vessel readiness (max {vessel_ready_wait_max}s)...")
            for i in range(vessel_ready_wait_max * 2): # Check twice per second
                try:
                    vessel = sc.active_vessel
                    # Try accessing a property that requires the vessel to be loaded
                    if vessel is not None and vessel.name: # Check name property exists
                         print(f"    Active vessel '{vessel.name}' found and responsive.")
                         vessel_found = True; break # Exit loop if vessel found
                except (krpc.error.RPCError, AttributeError, TypeError): pass # Ignore errors while waiting
                print(f"      Waiting for vessel... ({i/2.0 + 0.5:.1f}s)\r", end='', flush=True); time.sleep(0.5)

            if not vessel_found:
                print("\n!! ERROR: Failed to get responsive active vessel after quickload and extended wait.")
                raise RuntimeError("Failed to get active vessel after quickload.")

            reload_success = True # Mark reload successful *only* if vessel is found
        except ConnectionRefusedError: print("!! ERROR: KSP Connection Refused during reload attempt.")
        except krpc.error.RPCError as e: print(f"!! ERROR: kRPC Error during reload: {e}")
        except RuntimeError as e: print(f"!! ERROR: {e}") # Catch vessel not found error
        except Exception as e: print(f"!! ERROR: Unexpected Error during reload: {type(e).__name__}: {e}")
        finally:
            if temp_conn:
                try: temp_conn.close(); print("    Reload connection closed.")
                except Exception: pass
        if not reload_success:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! RELOAD FAILED. Stopping script.                       !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            raise RuntimeError(f"Failed to reload quicksave '{QUICKSAVE_NAME}'. Stopping GA.")
        print(f"\n>>> Now training Generation {next_gen_num}... <<<")
        time.sleep(1)
    else:
         print(f"\n G----------------------------------------------------------G"); print("| Final generation complete. No reload needed.")


# --- Fitness Function (Corrected Situation Check, Cleanup, added Debug) ---
def fitness_func(ga_instance, solution, solution_idx):
    Kp, Ki, Kd = solution
    gen_num = ga_instance.generations_completed + 1
    print(f"\n--- Evaluating Solution {solution_idx} (Gen {gen_num}): Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} ---")

    conn = None; vessel = None; control = None; ap = None
    total_abs_heading_error = 0; total_vertical_speed = 0
    num_updates = 0; fitness = 0.0
    setup_successful = False; start_time = 0.0
    pid_controller = None

    try:
        # --- Step 1: Connect and Get Objects ---
        print("  [SETUP] Connecting to kRPC...")
        conn = krpc.connect(name=f'{KSP_CONNECTION_NAME}_Eval_{solution_idx}')
        print("  [SETUP] Getting SpaceCenter objects...")
        sc = conn.space_center
        vessel = sc.active_vessel
        if vessel is None: raise RuntimeError("Vessel object is None after connect!")
        # <<< FIXED: Get Situation Enum from space_center >>>
        VS = sc.VesselSituation # Get the VesselSituation enum class
        vessel_sit = vessel.situation # Get the vessel's current situation (enum member)
        print(f"  [SETUP] Vessel found: {vessel.name} (Status: {vessel_sit})")
        control = vessel.control; ap = vessel.auto_pilot
        surface_ref = vessel.surface_reference_frame
        flight = vessel.flight(surface_ref)
        print("  [SETUP] SpaceCenter objects obtained.")

        # --- Step 2 & 3: Initial State Setup & Pitch Control ---
        try:
            print(f"  [SETUP] Initial check: Altitude={flight.mean_altitude:.1f}, VSI={flight.vertical_speed:.2f}, Speed={flight.speed:.1f}")
            # <<< FIXED: Compare with correct enum members >>>
            if vessel_sit == VS.pre_launch or \
               vessel_sit == VS.landed or \
               vessel_sit == VS.splashed:
                raise RuntimeError(f"Vessel is not flying! Situation: {vessel_sit}")

            print(f"  [SETUP] Initializing PID Controller...")
            pid_controller = PID(Kp, Ki, Kd, TARGET_HEADING)
            pid_controller.reset()

            print(f"  [SETUP] Setting initial throttle: {INITIAL_THROTTLE}")
            control.throttle = INITIAL_THROTTLE
            print("  [SETUP] Ensuring SAS/AP are disabled before applying eval method...")
            control.sas = False
            if ap is not None and hasattr(ap, 'disengage'): ap.disengage()
            else: print("  [SETUP] Warning: Autopilot object invalid or lacks disengage method.")
            time.sleep(0.2)
            sas_state_after_disable = control.sas; ap_state_after_disable = "N/A"
            try:
                 if hasattr(ap, 'sas_mode'): ap_state_after_disable = f"SAS Mode: {ap.sas_mode}"
                 else: ap_state_after_disable = f"AP State Check N/A"
            except Exception as ap_check_e: ap_state_after_disable = f"Error checking AP state: {ap_check_e}"
            print(f"  [SETUP] State before eval method: SAS={sas_state_after_disable}, {ap_state_after_disable}")

            # --- Apply Pitch/Roll Control Method ---
            if PITCH_CONTROL_METHOD == 'AP_PITCH':
                print(f"  [SETUP] Engaging Autopilot: Target Pitch={TARGET_INITIAL_PITCH} deg, Target Roll=0 deg")
                ap.reference_frame = surface_ref; ap.target_pitch = TARGET_INITIAL_PITCH; ap.target_roll = 0
                if not hasattr(ap, 'engage'): raise AttributeError("Autopilot object lacks engage method")
                ap.engage()
                stabilization_wait = 2.0; print(f"  [SETUP] Waiting {stabilization_wait}s for initial AP set..."); time.sleep(stabilization_wait)
                ap_pitch_after_engage = "N/A";
                try: ap_pitch_after_engage = f"{ap.target_pitch:.1f}"
                except: pass
                print(f"  [SETUP] AP state after wait: TargetPitch={ap_pitch_after_engage}")
            elif PITCH_CONTROL_METHOD == 'SAS_STABILITY':
                 # Corrected SAS check for this method
                 print("  [SETUP] Engaging SAS (Stability Assist) for evaluation...")
                 control.sas = True; time.sleep(0.1)
                 if control.sas: # Check if it actually enabled
                     print("  [SETUP] SAS engaged successfully for evaluation.")
                     try:
                          if hasattr(conn.space_center, 'SASMode'): ap.sas_mode = conn.space_center.SASMode.stability_assist; print("  [SETUP] SAS Mode set to Stability Assist.")
                          else: print("  [SETUP] (SASMode attribute not found, basic SAS enable)")
                          stabilization_wait = 2.0; print(f"  [SETUP] Waiting {stabilization_wait}s for initial SAS set..."); time.sleep(stabilization_wait)
                          print(f"  [SETUP] SAS status after wait: {control.sas}")
                     except krpc.error.RPCError as e: print(f"  [SETUP] Warning: Could not set SAS mode: {e}. Relying on basic SAS enable."); time.sleep(1)
                 else: raise RuntimeError("SAS unavailable or failed to engage for SAS_STABILITY method")
            else: raise RuntimeError(f"Unknown PITCH_CONTROL_METHOD: {PITCH_CONTROL_METHOD}")

            print("  [SETUP] Setup phase completed successfully.")
            setup_successful = True
        # Catch errors during setup phase
        except (krpc.error.RPCError, RuntimeError, AttributeError, TypeError) as setup_e:
            print(f"!! FATAL ERROR during evaluation SETUP ({type(setup_e).__name__}): {setup_e}");
            # Only print traceback for non-Runtime errors which might be more unexpected
            if not isinstance(setup_e, RuntimeError):
                 import traceback; traceback.print_exc();
        except Exception as setup_e: # Catch any other unexpected errors during setup
            print(f"!! FATAL UNEXPECTED ERROR during evaluation SETUP: {type(setup_e).__name__}: {setup_e}"); import traceback; traceback.print_exc()


        # --- Step 4: Simulation Loop ---
        if setup_successful:
            print(f"  [LOOP] Starting loop, duration={SIMULATION_DURATION}s...")
            start_time = time.time()
            last_print_time = start_time
            loop_error = None; iteration_count = 0
            try:
                print("  [LOOP] Entering simulation loop...")
                while time.time() - start_time < SIMULATION_DURATION:
                    iteration_count += 1
                    try: current_heading = flight.heading; current_vsi = flight.vertical_speed
                    except krpc.error.RPCError as read_e: print(f"!! kRPC Error reading flight data (Iteration {iteration_count}): {read_e}"); loop_error = read_e; break
                    num_updates += 1
                    heading_error = TARGET_HEADING - current_heading
                    while heading_error > 180: heading_error -= 360
                    while heading_error <= -180: heading_error += 360
                    total_abs_heading_error += abs(heading_error); total_vertical_speed += current_vsi
                    if pid_controller is None: raise RuntimeError("PID Controller not initialized before loop!")
                    yaw_input = pid_controller.update(current_heading)
                    try: control.yaw = yaw_input
                    except krpc.error.RPCError as write_e: print(f"!! kRPC Error setting control.yaw (Iteration {iteration_count}): {write_e}"); loop_error = write_e; break
                    if time.time() - last_print_time > 3.0: print(f"    t={time.time()-start_time:.1f}s, Head={current_heading:.1f}, Err={heading_error:.1f}, Yaw={yaw_input:.2f}, VSI={current_vsi:.4f}"); last_print_time = time.time() # Using .4f for VSI
                    time.sleep(CONTROL_UPDATE_INTERVAL)
                elapsed_time = time.time() - start_time
                if loop_error is None: print(f"  [LOOP] Simulation loop completed normally after {elapsed_time:.2f}s and {iteration_count} iterations.")
                else: print(f"  [LOOP] Simulation loop exited due to error after {elapsed_time:.2f}s and {iteration_count} iterations.")
            except Exception as loop_e: print(f"!! Unexpected Error INSIDE simulation loop structure (Iteration {iteration_count}): {type(loop_e).__name__}: {loop_e}"); loop_error = loop_e

            # --- Step 5 & 6: Cleanup & Fitness Calc ---
            print("  Simulation loop processing finished.")
            try: print("  Resetting yaw control after loop...");
            if 'control' in locals() and control is not None: control.yaw = 0
            except (krpc.error.RPCError, AttributeError): print("  Error resetting yaw after loop.")
            if num_updates > 0 and loop_error is None:
                average_abs_heading_error = total_abs_heading_error / num_updates; average_vertical_speed = total_vertical_speed / num_updates
                vsi_deviation = abs(average_vertical_speed - TARGET_VERTICAL_SPEED)
                if average_vertical_speed < (TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION): print(f"    Applying diving penalty (Avg VSI {average_vertical_speed:.2f} < {TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION:.2f})"); vsi_component_penalty = vsi_deviation * DIVING_PENALTY_MULTIPLIER
                else: vsi_component_penalty = vsi_deviation
                combined_error = (HEADING_WEIGHT * average_abs_heading_error) + (VSI_WEIGHT * vsi_component_penalty); fitness = 1.0 / (combined_error + 1e-6)
                print(f"  Avg Abs Heading Error: {average_abs_heading_error:.4f}"); print(f"  Average Vertical Speed: {average_vertical_speed:.4f} (Target: {TARGET_VERTICAL_SPEED})"); print(f"  VSI Deviation (Penalized): {vsi_component_penalty:.4f}"); print(f"  Combined Weighted Error: {combined_error:.4f} -> Fitness: {fitness:.4f}")
            elif loop_error is not None: print(f"  Simulation loop interrupted by error. Assigning low fitness."); fitness = 0.0
            else: print("  No updates occurred during simulation loop, fitness set to 0."); fitness = 0.0
        else: print("  Skipping simulation loop because setup failed."); fitness = 0.0

    # Catch errors from the very beginning
    except ConnectionRefusedError: print("!! KSP Connection Refused."); fitness = 0.0
    except (krpc.error.RPCError, RuntimeError, AttributeError, TypeError) as e: print(f"!! Error during initial connect/setup ({type(e).__name__}): {e}"); fitness = 0.0
    except Exception as e: print(f"!! Unexpected Error during initial connect/setup: {type(e).__name__}: {e}"); import traceback; traceback.print_exc(); fitness = 0.0
    finally:
        # --- Step 7: Final Cleanup (Corrected AP disengage) ---
        if conn:
            print("  Cleaning up controls and connection after evaluation...")
            try:
                if 'control' in locals() and control is not None:
                    try: control.yaw = 0
                    except krpc.error.RPCError: pass
                    if PITCH_CONTROL_METHOD == 'SAS_STABILITY':
                        try: control.sas = False
                        except krpc.error.RPCError: pass
                if 'ap' in locals() and ap is not None and PITCH_CONTROL_METHOD == 'AP_PITCH':
                    try:
                        # <<< FIXED: Call disengage directly >>>
                        ap.disengage()
                        print("  Autopilot disengaged (cleanup).")
                    except (krpc.error.RPCError, AttributeError): pass # Ignore errors disengaging
            except Exception as e_clean: print(f"  Unexpected error during control cleanup: {e_clean}")
            try: conn.close(); print("  kRPC connection closed.")
            except Exception: print("  Error closing kRPC connection."); pass

    print(f"  Fitness returned for Sol {solution_idx} (Gen {gen_num}): {fitness:.4f}")
    return fitness

# --- Genetic Algorithm Setup ---
gene_space = [{'low': 0.0, 'high': 2.5}, {'low': 0.0, 'high': 2.5}, {'low': 0.0, 'high': 2.5}]
num_genes = len(gene_space)
# Using GA parameters from top

print("\n=== Initializing Genetic Algorithm ===")
ga_instance = pygad.GA(num_generations=num_generations, num_parents_mating=num_parents_mating, fitness_func=fitness_func, sol_per_pop=sol_per_pop, num_genes=num_genes, gene_space=gene_space, parent_selection_type=parent_selection_type, keep_parents=keep_parents, mutation_type=mutation_type, mutation_percent_genes=mutation_percent_genes, allow_duplicate_genes=False, on_generation=generation_reload_callback) # Using reload callback

# --- Run the GA ---
print("\n=== Starting Genetic Algorithm Run ===")
print(f"Target Heading: {TARGET_HEADING} degrees"); print(f"Target Vertical Speed: {TARGET_VERTICAL_SPEED} m/s"); print(f"Simulation Duration per Solution: {SIMULATION_DURATION} seconds"); print(f"Pitch Control Method during Eval: {PITCH_CONTROL_METHOD}")
if PITCH_CONTROL_METHOD == 'AP_PITCH': print(f"Target Initial Pitch during Eval: {TARGET_INITIAL_PITCH} degrees")
print("\n!!! IMPORTANT !!!"); print(f"Ensure quicksave '{QUICKSAVE_NAME}' exists."); print("Script will reload between generations.")
input("Press Enter to start...")
try: ga_instance.run()
except KeyboardInterrupt: print("\n!!! GA Run Interrupted !!!")
except RuntimeError as e: print(f"\n!!! GA Run Stopped due to Error: {e} !!!")
except Exception as e: print(f"\n!!! UNEXPECTED ERROR: {type(e).__name__}: {e} !!!"); import traceback; traceback.print_exc()

# --- Results ---
print("\n=== Genetic Algorithm Finished ===")
if ga_instance.generations_completed > 0:
    try:
        print("  Generating fitness plot...")
        # <<< FIXED: Correct plotting call >>>
        ga_instance.plot_fitness()
    except ImportError: print("\nInstall matplotlib to see plot.")
    except AttributeError: print("\nWarning: PyGAD version might not support plot_fitness().")
    except Exception as e: print(f"\nCould not display plot: {type(e).__name__}: {e}")
    try:
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("\nBest solution found:");
        if hasattr(ga_instance, 'best_solution_generation'): print(f"  Generation: {ga_instance.best_solution_generation}")
        print(f"  Index in Last Pop: {solution_idx}"); print(f"  Gains (Kp, Ki, Kd): ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})"); print(f"  Fitness value = {solution_fitness:.6f}")
    except Exception as e: print(f"\nError retrieving best solution details: {e}")
else: print("\nNo generations were completed.")
print("\nOptimization complete.")