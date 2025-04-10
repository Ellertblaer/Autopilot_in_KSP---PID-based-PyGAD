# Required Libraries: pip install krpc pygad numpy matplotlib
import krpc
import time
import math
import pygad
import numpy as np
import matplotlib.pyplot as plt # Needed for ga_instance.plot_fitness()

# --- Configuration ---
KSP_CONNECTION_NAME = 'GA_PID_Tuner_SAS_Only' # Updated name
# QUICKSAVE_NAME = "midair - hærra" # <<< REMOVED - Not reloading anymore >>>

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

# --- Callback Function for Inter-Generation SAS Enable (MODIFIED) ---
def generation_sas_callback(ga_instance):
    """
    Called by PyGAD after each generation. Prints status, enables SAS
    (Stability Assist) on the active vessel, and announces the start of the next generation.
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

    # Enable SAS before starting the next generation (except after the last one)
    if completed_gen < ga_instance.num_generations:
        print(f"| Preparing for Generation {next_gen_num}: Enabling SAS")
        print(f" G----------------------------------------------------------G")

        sas_enabled_successfully = False
        temp_conn = None
        try:
            # --- Step 1: Connect and Get Vessel ---
            print("    Connecting to kRPC to enable SAS...")
            # Use a distinct connection name for this task
            temp_conn = krpc.connect(name=f'{KSP_CONNECTION_NAME}_SAS_Enabler_{next_gen_num}')
            sc = temp_conn.space_center
            vessel = sc.active_vessel
            if vessel is None:
                 raise RuntimeError("Could not get active vessel.")
            print(f"    Active vessel: {vessel.name}")
            control = vessel.control
            ap = vessel.auto_pilot

            # --- Step 2: Enable SAS ---
            print("    Attempting to enable SAS...")
            if control.sas_available:
                control.sas = True
                time.sleep(0.3) # Pause for SAS to engage
                if control.sas:
                    print("    SAS Enabled successfully.")
                    sas_enabled_successfully = True
                    # Optionally set mode to stability assist
                    try:
                        if hasattr(sc, 'SASMode'):
                            ap.sas_mode = sc.SASMode.stability_assist
                            print("    SAS Mode set to Stability Assist.")
                        else: print("    (SASMode attribute not found, basic SAS enabled)")
                    except krpc.error.RPCError as sas_mode_e: print(f"    Warning: Could not set SAS mode: {sas_mode_e}")
                else: print("!! WARNING: Attempted to enable SAS, but it did not engage.")
            else: print("!! WARNING: SAS is not available on this vessel.")

        except ConnectionRefusedError: print("!! ERROR: KSP Connection Refused during SAS attempt."); sas_enabled_successfully = False
        except krpc.error.RPCError as e: print(f"!! ERROR: kRPC Error during SAS attempt: {e}"); sas_enabled_successfully = False
        except RuntimeError as e: print(f"!! ERROR: {e}"); sas_enabled_successfully = False # Catch vessel error
        except Exception as e: print(f"!! ERROR: Unexpected Error during SAS attempt: {type(e).__name__}: {e}"); sas_enabled_successfully = False
        finally:
            if temp_conn:
                try: temp_conn.close(); print("    SAS connection closed.")
                except Exception: pass

        # --- Step 3: Check Success and Proceed ---
        # Decide if SAS failure should stop the script. For now, just warn.
        if not sas_enabled_successfully:
            print("!! WARNING: Failed to enable SAS between generations. Evaluations might start less stable.")
            # If you want to stop on failure, uncomment the line below:
            # raise RuntimeError(f"Failed during SAS setup for Generation {next_gen_num}.")

        # Announce next generation regardless of SAS success (unless stopping)
        print(f"\n>>> Now training Generation {next_gen_num}... <<<")
        time.sleep(1) # Small pause before evaluations begin

    else:
         print(f"\n G----------------------------------------------------------G")
         print("| Final generation complete. No SAS enable needed.")


# --- Fitness Function (Unchanged from previous robust version) ---
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
                      else: print("  (SASMode attribute not found, relying on basic SAS enable)")
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
        loop_error = None

        try:
            while time.time() - start_time < SIMULATION_DURATION:
                current_heading = flight.heading
                current_vsi = flight.vertical_speed
                num_updates += 1

                heading_error = TARGET_HEADING - current_heading
                while heading_error > 180: heading_error -= 360
                while heading_error <= -180: heading_error += 360
                total_abs_heading_error += abs(heading_error)

                yaw_input = pid_controller.update(current_heading)
                control.yaw = yaw_input

                total_vertical_speed += current_vsi

                if time.time() - last_print_time > 3.0:
                     print(f"    t={time.time()-start_time:.1f}s, Head={current_heading:.1f}, Err={heading_error:.1f}, Yaw={yaw_input:.2f}, VSI={current_vsi:.4f}")
                     last_print_time = time.time()

                time.sleep(CONTROL_UPDATE_INTERVAL)
        except krpc.error.RPCError as loop_e: print(f"!! kRPC Error INSIDE simulation loop: {loop_e}"); loop_error = loop_e
        except Exception as loop_e: print(f"!! Unexpected Error INSIDE simulation loop: {type(loop_e).__name__}: {loop_e}"); loop_error = loop_e

        # --- Simulation End & Cleanup ---
        print("  Simulation loop finished.")
        try: control.yaw = 0
        except krpc.error.RPCError: print("  kRPC error resetting yaw after loop.")

        # --- Calculate Fitness ---
        if num_updates > 0 and loop_error is None:
            average_abs_heading_error = total_abs_heading_error / num_updates
            average_vertical_speed = total_vertical_speed / num_updates
            vsi_deviation = abs(average_vertical_speed - TARGET_VERTICAL_SPEED)

            if average_vertical_speed < (TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION):
                print(f"    Applying diving penalty (Avg VSI {average_vertical_speed:.2f} < {TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION:.2f})")
                vsi_component_penalty = vsi_deviation * DIVING_PENALTY_MULTIPLIER
            else: vsi_component_penalty = vsi_deviation

            combined_error = (HEADING_WEIGHT * average_abs_heading_error) + (VSI_WEIGHT * vsi_component_penalty)
            fitness = 1.0 / (combined_error + 1e-6)

            print(f"  Avg Abs Heading Error: {average_abs_heading_error:.4f}")
            print(f"  Average Vertical Speed: {average_vertical_speed:.4f} (Target: {TARGET_VERTICAL_SPEED})")
            print(f"  VSI Deviation (Penalized): {vsi_component_penalty:.4f}")
            print(f"  Combined Weighted Error: {combined_error:.4f} -> Fitness: {fitness:.4f}")

        elif loop_error is not None: print(f"  Simulation loop interrupted by error. Assigning low fitness."); fitness = 0.0
        else: print("  No updates occurred during simulation loop, fitness set to 0."); fitness = 0.0

    except ConnectionRefusedError: print("!! KSP Connection Refused."); fitness = 0.0
    except krpc.error.RPCError as e: print(f"!! kRPC Error during setup/eval: {e}"); fitness = 0.0
    except Exception as e:
        print(f"!! Unexpected Error during setup/eval: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc(); fitness = 0.0
    finally:
        if conn:
            try:
                print("  Cleaning up controls after evaluation...")
                if 'control' in locals() and control is not None:
                    control.yaw = 0
                    if PITCH_CONTROL_METHOD == 'SAS_STABILITY': control.sas = False
                if 'ap' in locals() and ap is not None and PITCH_CONTROL_METHOD == 'AP_PITCH':
                     if ap.engaged: ap.disengage()
            except krpc.error.RPCError: print("  kRPC error during final cleanup.")
            except Exception as e_clean: print(f"  Unexpected error during final cleanup: {e_clean}")
            finally:
                 try: conn.close(); print("  Closing kRPC connection for evaluation.")
                 except Exception: pass

    print(f"  Fitness returned for Sol {solution_idx} (Gen {gen_num}): {fitness:.4f}")
    return fitness

# --- Genetic Algorithm Setup ---
# (Gene Space and GA Parameters remain the same)
gene_space = [
    {'low': 0.0, 'high': 2.5}, {'low': 0.0, 'high': 2.5}, {'low': 0.0, 'high': 2.5}
]
num_genes = len(gene_space)
num_generations = 6
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
                       # vvv Assign the NEW callback function vvv
                       on_generation=generation_sas_callback
                       # ^^^ Assign the NEW callback function ^^^
                       )

# --- Run the GA ---
print("\n=== Starting Genetic Algorithm Run ===")
print(f"Target Heading: {TARGET_HEADING} degrees")
print(f"Target Vertical Speed: {TARGET_VERTICAL_SPEED} m/s")
print(f"Simulation Duration per Solution: {SIMULATION_DURATION} seconds")
print(f"Pitch Control Method during Eval: {PITCH_CONTROL_METHOD}")
if PITCH_CONTROL_METHOD == 'AP_PITCH': print(f"Target Initial Pitch during Eval: {TARGET_INITIAL_PITCH} degrees")
# MODIFIED Print statement - removed mention of quicksave
print("\nThe script will attempt to enable SAS between generations.")
print("Ensure KSP is running and the vessel is in a controllable state.")
input("Press Enter to start the GA optimization...")

try:
    ga_instance.run()
except KeyboardInterrupt: print("\n!!! GA Run Interrupted by User !!!")
except RuntimeError as e: print(f"\n!!! GA Run Stopped due to Error: {e} !!!") # Catch potential SAS failure if uncommented
except Exception as e:
     print(f"\n!!! UNEXPECTED ERROR during GA run: {type(e).__name__}: {e} !!!")
     import traceback; traceback.print_exc()


# --- Results ---
print("\n=== Genetic Algorithm Finished ===")
# (Plotting and results section uses the fixed version)
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