# Required Libraries: pip install krpc pygad numpy matplotlib
import krpc
import time
import math
import pygad
import numpy as np
import matplotlib.pyplot as plt # Needed for ga_instance.plot_fitness()

#STILLINGAR, ÆLA ÆLA ÆLA
# --- Configuration ---
KSP_CONNECTION_NAME = 'GA_PID_Tuner_Heading_VSI'
TARGET_HEADING = 45      # Target heading in degrees
TARGET_VERTICAL_SPEED = 1.0 # Target vertical speed in m/s (e.g., 1.0 for a gentle climb)
SIMULATION_DURATION = 20 # How many seconds to test each set of PID gains (Increased slightly for VSI stability)
CONTROL_UPDATE_INTERVAL = 0.05 # How often to update PID and controls (seconds) - slightly faster updates might help
INITIAL_THROTTLE = 1.0   # Initial throttle setting for the test flight

conn = krpc.connect(name='Til að control virki')
vessel = conn.space_center.active_vessel
control = vessel.control
ap = vessel.auto_pilot

# --- Fitness Function Weights (TUNE THESE!) ---
# How much to prioritize heading accuracy vs vertical speed accuracy
# Higher weight means that aspect is more important for the fitness score.
HEADING_WEIGHT = 1.0
VSI_WEIGHT = 0.5 # Start with VSI being less critical than heading, adjust as needed
# Penalty for diving significantly below the target VSI
DIVING_PENALTY_MULTIPLIER = 3.0 # Increase penalty if average VSI is well below target
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

# --- Callback Function for Inter-Generation Pause (Unchanged) ---
def generation_countdown_callback(ga_instance):
    """
    Called by PyGAD after each generation. Prints status, provides a pause,
    and announces the start of the next generation.
    """
    completed_gen = ga_instance.generations_completed
    next_gen_num = completed_gen + 1

    #Kemur beint á undan tékkinu
    control.sas = True 

    print(f"\n G----------------------------------------------------------G")
    print(f"| Generation {completed_gen} Finished.")
    try:
        last_gen_fitness = ga_instance.last_generation_fitness
        if last_gen_fitness is not None and len(last_gen_fitness) > 0:
             best_fitness_this_gen = np.max(last_gen_fitness)
             print(f"| Best Fitness in Generation {completed_gen}: {best_fitness_this_gen:.4f}")
        else:
             print("| (Could not retrieve last generation fitness)")
    except Exception as e:
        print(f"| (Error retrieving fitness: {e})")

    if completed_gen < ga_instance.num_generations:
 
        print(f" G----------------------------------------------------------G")

        countdown_duration = 15
        for i in range(countdown_duration, 0, -1):
            print(f"    Prep time remaining: {i:2d} seconds... \r", end='', flush=True)
            time.sleep(1)
        print("                                        \r", end='')

        print(f"\n>>> Now training Generation {next_gen_num}... <<<")
        time.sleep(1)
    else:
         print(f" G----------------------------------------------------------G")
         print("| Final generation complete.")


# --- Fitness Function (MODIFIED) ---
def fitness_func(ga_instance, solution, solution_idx):
    """
    Evaluates a set of PID gains (solution) by running a KSP simulation segment.
    Fitness is based on minimizing heading error AND minimizing deviation
    from the TARGET_VERTICAL_SPEED, penalizing diving. Higher fitness is better.
    """
    Kp, Ki, Kd = solution

    print(f"\n--- Evaluating Solution {solution_idx} (Gen {ga_instance.generations_completed+1}): Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} ---")

    conn = None
    total_abs_heading_error = 0
    total_vertical_speed = 0 # Accumulate VSI for averaging
    num_updates = 0
    fitness = 0.0

    try:
        # Connect to KSP
        print("  Connecting to kRPC...")
        conn = krpc.connect(name=f'{KSP_CONNECTION_NAME}_Eval_{solution_idx}')
        vessel = conn.space_center.active_vessel
        control = vessel.control
        ap = vessel.auto_pilot
        # Using surface reference frame is generally better for altitude/VSI relative to ground
        surface_ref = vessel.surface_reference_frame
        flight = vessel.flight(surface_ref) # Use surface flight info

        # --- Simulation Setup ---
        print(f"  Setting up simulation: Target Head={TARGET_HEADING}, Target VSI≈{TARGET_VERTICAL_SPEED}, Dur={SIMULATION_DURATION}s")
        pid_controller = PID(Kp, Ki, Kd, TARGET_HEADING)
        pid_controller.reset()

        # Prepare vessel state
        print(f"  Setting initial throttle: {INITIAL_THROTTLE}")
        control.throttle = INITIAL_THROTTLE

        # --- Apply Pitch/Roll Control Method ---
        # This section attempts to set an initial pitch/roll state.
        # The strict VSI check is removed; fitness function now handles VSI performance.
        if PITCH_CONTROL_METHOD == 'AP_PITCH':
            print(f"  Engaging Autopilot: Target Pitch={TARGET_INITIAL_PITCH} deg, Target Roll=0 deg")
            ap.reference_frame = surface_ref # Use surface frame for AP pitch/roll
            ap.target_pitch = TARGET_INITIAL_PITCH
            ap.target_roll = 0
            ap.engage()
            stabilization_wait = 2.0 # Reduced wait, fitness judges stability now
            print(f"  Waiting {stabilization_wait}s for initial AP set...")
            time.sleep(stabilization_wait)
        elif PITCH_CONTROL_METHOD == 'SAS_STABILITY':
             if vessel.control.sas_available:
                 print("  Engaging SAS (Stability Assist)...")
                 control.sas = True
                 time.sleep(0.1)
                 try:
                      if hasattr(conn.space_center, 'SASMode'):
                          ap.sas = conn.space_center.SASMode.stability_assist
                          print("  SAS Mode set to Stability Assist.")
                      else:
                          print("  (SASMode attribute not found, relying on basic SAS enable)")
                      stabilization_wait = 2.0 # Reduced wait
                      print(f"  Waiting {stabilization_wait}s for initial SAS set...")
                      time.sleep(stabilization_wait)
                 except krpc.error.RPCError as e:
                      print(f"  Warning: Could not set SAS mode: {e}. Relying on basic SAS enable.")
                      control.sas._mode = True; time.sleep(1)
             else:
                 print("!! Warning: SAS unavailable for SAS_STABILITY method! Aborting solution.")
                 return 0.0
        else:
             print(f"!! Error: Unknown PITCH_CONTROL_METHOD '{PITCH_CONTROL_METHOD}'. Aborting solution.")
             return 0.0

        # --- REMOVED Strict Pre-Simulation Stability Check ---
        print("  Proceeding to simulation loop (Fitness function will evaluate VSI).")

        # --- Simulation Loop ---
        print("  Starting PID control loop...")
        start_time = time.time()
        last_print_time = start_time

        while time.time() - start_time < SIMULATION_DURATION:
            # Get current state
            current_heading = flight.heading
            current_vsi = flight.vertical_speed
            num_updates += 1

            # --- Heading Control ---
            heading_error = TARGET_HEADING - current_heading
            while heading_error > 180: heading_error -= 360
            while heading_error <= -180: heading_error += 360
            total_abs_heading_error += abs(heading_error)

            # Update PID and apply Yaw control
            yaw_input = pid_controller.update(current_heading)
            control.yaw = yaw_input

            # --- Accumulate VSI ---
            total_vertical_speed += current_vsi

            # Optional: Print status periodically
            if time.time() - last_print_time > 3.0: # Print more frequently if needed
                 print(f"    t={time.time()-start_time:.1f}s, Head={current_heading:.1f}, Err={heading_error:.1f}, Yaw={yaw_input:.2f}, VSI={current_vsi:.1f}")
                 last_print_time = time.time()

            time.sleep(CONTROL_UPDATE_INTERVAL)

        # --- Simulation End & Cleanup ---
        print("  Simulation finished.")
        control.yaw = 0 # Reset yaw control first

        # --- Calculate Fitness (MODIFIED) ---
        if num_updates > 0:
            # Calculate average heading error
            average_abs_heading_error = total_abs_heading_error / num_updates

            # Calculate average vertical speed
            average_vertical_speed = total_vertical_speed / num_updates

            # Calculate deviation from target vertical speed
            vsi_deviation = abs(average_vertical_speed - TARGET_VERTICAL_SPEED)

            # Apply heavy penalty if diving significantly
            if average_vertical_speed < (TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION):
                print(f"    Applying diving penalty (Avg VSI {average_vertical_speed:.2f} < {TARGET_VERTICAL_SPEED - VSI_ALLOWED_NEGATIVE_DEVIATION:.2f})")
                vsi_component_penalty = vsi_deviation * DIVING_PENALTY_MULTIPLIER
            else:
                vsi_component_penalty = vsi_deviation # Standard deviation otherwise

            # Combine errors using weights
            # Lower combined error is better
            combined_error = (HEADING_WEIGHT * average_abs_heading_error) + \
                             (VSI_WEIGHT * vsi_component_penalty)

            # Fitness is inverse of combined error (higher fitness is better)
            fitness = 1.0 / (combined_error + 1e-6) # Add epsilon to avoid division by zero

            print(f"  Avg Abs Heading Error: {average_abs_heading_error:.4f}")
            print(f"  Average Vertical Speed: {average_vertical_speed:.4f} (Target: {TARGET_VERTICAL_SPEED})")
            print(f"  VSI Deviation (Penalized): {vsi_component_penalty:.4f}")
            print(f"  Combined Weighted Error: {combined_error:.4f} -> Fitness: {fitness:.4f}")

        else:
            print("  No updates occurred, fitness set to 0.")
            fitness = 0.0

    except ConnectionRefusedError:
        print("!! KSP Connection Refused. Is KSP running with kRPC server active? Skipping solution.")
        return 0
    except krpc.error.RPCError as e:
        print(f"!! kRPC Error during simulation: {e}. Skipping solution.")
        return 0
    except Exception as e:
        print(f"!! Unexpected Error during simulation: {type(e).__name__}: {e}. Skipping solution.")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # --- Ensure Controls are Reset and Connection Closed ---
        if conn:
            try:
                print("  Cleaning up controls...")
                if 'control' in locals():
                    control.yaw = 0
                    # Maybe reduce throttle slightly after test?
                    # control.throttle = 0.8
                    if PITCH_CONTROL_METHOD == 'SAS_STABILITY':
                        control.sas = False
                if 'ap' in locals() and PITCH_CONTROL_METHOD == 'AP_PITCH':
                     if ap.engaged:
                          ap.disengage()
            except krpc.error.RPCError:
                 print("  kRPC error during cleanup (likely disconnected).")
            except Exception as e_clean:
                 print(f"  Unexpected error during cleanup: {e_clean}")
            finally:
                 print("  Closing kRPC connection.")
                 conn.close()

    return fitness

# --- Genetic Algorithm Setup ---

# Gene Space: Define the possible range for each gene (Kp, Ki, Kd)
# --- !!! THESE RANGES ARE CRITICAL - Adjust based on trial-and-error !!! ---
gene_space = [
    {'low': 0.0, 'high': 2.5},  # Range for Kp
    {'low': 0.0, 'high': 2.5},  # Range for Ki
    {'low': 0.0, 'high': 2.5}   # Range for Kd
]
num_genes = len(gene_space)

#STILLINGAR, ÆLA ÆLA ÆLA
# --- GA Parameters - TUNE THESE! ---
# Using the faster parameters from previous version
num_generations = 10
num_parents_mating = 5
sol_per_pop = 15
parent_selection_type = "sss"
keep_parents = 2
mutation_type = "random"
mutation_percent_genes = 30

print("\n=== Initializing Genetic Algorithm ===")
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func, # Using the modified fitness function
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       allow_duplicate_genes=False,
                       on_generation=generation_countdown_callback
                       )

# --- Run the GA ---
print("\n=== Starting Genetic Algorithm Run ===")
print(f"Target Heading: {TARGET_HEADING} degrees")
print(f"Target Vertical Speed: {TARGET_VERTICAL_SPEED} m/s")
print(f"Simulation Duration per Solution: {SIMULATION_DURATION} seconds")
print(f"Pitch Control Method: {PITCH_CONTROL_METHOD}")
if PITCH_CONTROL_METHOD == 'AP_PITCH':
    print(f"Target Initial Pitch: {TARGET_INITIAL_PITCH} degrees")
print("Ensure KSP is running, kRPC server is active,")
print("and the aircraft is in a relatively stable flight state (airborne, decent speed).")
print("The script will optimize for BOTH heading and maintaining target vertical speed.")
print("Use the 15s pause between generations to reset the aircraft if needed.")
input("Press Enter to start the GA optimization...")

try:
    ga_instance.run()
except KeyboardInterrupt:
    print("\n!!! GA Run Interrupted by User !!!")

# --- Results ---
print("\n=== Genetic Algorithm Finished ===")

if ga_instance.generations_completed > 0:
    try:
        fig, ax = plt.subplots()
        ga_instance.plot_fitness(plot_type="plot", ax=ax)
        plt.title("GA Fitness Progression")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\nInstall matplotlib (pip install matplotlib) to see the fitness plot.")
    except Exception as e:
        print(f"\nCould not display plot: {e}")

    try:
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("\nBest solution found:")
        if hasattr(ga_instance, 'best_solution_generation'):
            print(f"  Generation: {ga_instance.best_solution_generation}")
        print(f"  Index in Last Pop: {solution_idx}")
        print(f"  Gains (Kp, Ki, Kd): ({solution[0]:.6f}, {solution[1]:.6f}, {solution[2]:.6f})")
        print(f"  Fitness value = {solution_fitness:.6f}")
    except Exception as e:
        print(f"\nError retrieving best solution details: {e}")
        print("  (Possibly GA run was too short or interrupted before completion)")

else:
    print("\nNo generations were completed (possibly interrupted early or failed).")

print("\nOptimization complete.")