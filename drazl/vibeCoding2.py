# Required Libraries: pip install krpc pygad numpy matplotlib
import krpc
import time
import math
import pygad
import numpy as np
import matplotlib.pyplot as plt # Needed for ga_instance.plot_fitness()

# --- Configuration ---
KSP_CONNECTION_NAME = 'GA_PID_Tuner'
TARGET_HEADING = 45      # Target heading in degrees
SIMULATION_DURATION = 10 # How many seconds to test each set of PID gains
CONTROL_UPDATE_INTERVAL = 0.01 # How often to update PID and controls (seconds)
INITIAL_THROTTLE = 1.0   # Initial throttle setting for the test flight
# --- Choose ONE Pitch/Altitude Control Method ---
# Options: 'AP_PITCH', 'SAS_STABILITY'
# 'SAS_STABILITY' often works better if AP struggles with pitch target
PITCH_CONTROL_METHOD = 'AP_PITCH'
TARGET_INITIAL_PITCH = 20.0 # Degrees (Only used if PITCH_CONTROL_METHOD = 'AP_PITCH') - 30 is very high, ensure aircraft can handle it!

# --- PID Controller Class ---
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
        if dt <= 0: # Avoid division by zero or negative dt if time goes backwards slightly
            # Return last calculated output if dt is not positive
            # Or consider returning 0 if it's the very first update?
            return self.output

        # Calculate heading error (handle 360/0 degree wrap-around)
        error = self.setpoint - measured_value
        # Normalize error to -180 to 180 range
        while error > 180: error -= 360
        while error <= -180: error += 360

        # Proportional term
        p_term = self.Kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        self._integral = max(self.integral_limits[0], min(self.integral_limits[1], self._integral))
        i_term = self.Ki * self._integral

        # Derivative term
        # Avoid division by zero was already handled by checking dt > 0
        derivative = (error - self._last_error) / dt
        d_term = self.Kd * derivative

        # Compute output and clamp
        output = p_term + i_term + d_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))

        # Store values for next iteration
        self._last_error = error
        self._last_time = current_time

        self.output = output # Store last output for reference if needed
        return output

    def reset(self):
        self._integral = 0
        self._last_error = 0
        self._last_time = time.time()
        self.output = 0

# --- Callback Function for Inter-Generation Pause ---
def generation_countdown_callback(ga_instance):
    """
    Called by PyGAD after each generation. Prints status, provides a pause,
    and announces the start of the next generation.
    """
    completed_gen = ga_instance.generations_completed
    next_gen_num = completed_gen + 1 # Calculate the number of the generation about to start

    print(f"\n G----------------------------------------------------------G")
    print(f"| Generation {completed_gen} Finished.")
    # Display best fitness for the generation that just finished
    try:
        last_gen_fitness = ga_instance.last_generation_fitness
        if last_gen_fitness is not None and len(last_gen_fitness) > 0:
             best_fitness_this_gen = np.max(last_gen_fitness)
             print(f"| Best Fitness in Generation {completed_gen}: {best_fitness_this_gen:.4f}")
        else:
             print("| (Could not retrieve last generation fitness)")
    except Exception as e:
        print(f"| (Error retrieving fitness: {e})")

    # Only provide countdown and announce next gen if not the final one
    if completed_gen < ga_instance.num_generations:
        print(f"| PREP TIME: You have 15 seconds to reset the aircraft state.")
        print(f" G----------------------------------------------------------G")

        countdown_duration = 15
        for i in range(countdown_duration, 0, -1):
            print(f"    Prep time remaining: {i:2d} seconds... \r", end='', flush=True)
            time.sleep(1)
        print("                                        \r", end='') # Clear the countdown line

        # <<< ANNOUNCE START OF NEXT GENERATION >>>
        print(f"\n>>> Now training Generation {next_gen_num}... <<<")
        time.sleep(1) # Small pause before evaluations begin
    else:
         print(f" G----------------------------------------------------------G")
         print("| Final generation complete.")


# --- Fitness Function (The core of the GA) ---
def fitness_func(ga_instance, solution, solution_idx):
    """
    Evaluates a set of PID gains (solution) by running a KSP simulation segment.
    Higher fitness is better.
    """
    Kp, Ki, Kd = solution # Unpack the gains from the GA solution

    # Note: The "Now training Generation X" message is printed by the callback *before* this function runs for the first solution of that generation.
    print(f"\n--- Evaluating Solution {solution_idx} (Gen {ga_instance.generations_completed+1}): Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} ---")

    conn = None
    total_abs_error = 0
    num_updates = 0
    fitness = 0.0 # Default fitness if simulation fails

    try:
        # Connect to KSP for this evaluation
        print("  Connecting to kRPC...")
        conn = krpc.connect(name=f'{KSP_CONNECTION_NAME}_Eval_{solution_idx}')
        vessel = conn.space_center.active_vessel
        control = vessel.control
        ap = vessel.auto_pilot
        # Consider vessel.surface_reference_frame if altitude/pitch relative to horizon matters more
        surface_ref = vessel.orbit.body.reference_frame
        flight = vessel.flight(surface_ref)

        # --- Simulation Setup ---
        print(f"  Setting up simulation: Target Heading={TARGET_HEADING}, Duration={SIMULATION_DURATION}s")
        pid_controller = PID(Kp, Ki, Kd, TARGET_HEADING)
        pid_controller.reset() # Ensure PID state is fresh

        # Prepare vessel state:
        print(f"  Setting initial throttle: {INITIAL_THROTTLE}")
        control.throttle = INITIAL_THROTTLE

        # --- Apply Pitch/Roll Control Method ---
        if PITCH_CONTROL_METHOD == 'AP_PITCH':
            print(f"  Engaging Autopilot: Target Pitch={TARGET_INITIAL_PITCH} deg, Target Roll=0 deg")
            ap.reference_frame = surface_ref
            ap.target_pitch = TARGET_INITIAL_PITCH
            ap.target_roll = 0
            ap.engage()
            stabilization_wait = 0.5 # Seconds to wait for AP to stabilize
            print(f"  Waiting {stabilization_wait}s for AP stabilization...")
            time.sleep(stabilization_wait)
        elif PITCH_CONTROL_METHOD == 'SAS_STABILITY':
             if vessel.control.sas_available:
                 print("  Engaging SAS (Stability Assist)...")
                 control.sas = True
                 time.sleep(0.1)
                 try:
                      if hasattr(conn.space_center, 'SASMode'):
                          ap.sas_mode = conn.space_center.SASMode.stability_assist
                          print("  SAS Mode set to Stability Assist.")
                      else:
                          print("  (SASMode attribute not found, relying on basic SAS enable)")
                      stabilization_wait = 3
                      print(f"  Waiting {stabilization_wait}s for SAS stabilization...")
                      time.sleep(stabilization_wait)
                 except krpc.error.RPCError as e:
                      print(f"  Warning: Could not set SAS mode: {e}. Relying on basic SAS enable.")
                      control.sas = True
                      time.sleep(1)
             else:
                 print("!! Warning: SAS unavailable for SAS_STABILITY method! Aborting solution.")
                 return 0.0
        else:
             print(f"!! Error: Unknown PITCH_CONTROL_METHOD '{PITCH_CONTROL_METHOD}'. Aborting solution.")
             return 0.0

        # --- Pre-Simulation Stability Check ---
        print("  Performing initial stability check (vertical speed > -2 m/s)...")
        stabilization_timeout = 10
        stabilization_start_time = time.time()
        initial_state_ok = False
        while time.time() - stabilization_start_time < stabilization_timeout:
            try:
                vertical_speed = flight.vertical_speed
                print(f"    Checking... VSI: {vertical_speed:.1f} m/s")
                if vertical_speed > -5.0:
                    print("  Initial state seems stable enough (not diving).")
                    initial_state_ok = True
                    break
            except krpc.error.RPCError as e:
                print(f"    kRPC error during stability check: {e}. Aborting solution.")
                if conn: conn.close()
                return 0.0
            time.sleep(0.5)

        if not initial_state_ok:
            print("!! Warning: Failed to stabilize initial vertical speed within timeout. Aborting solution.")
            control.yaw = 0
            if PITCH_CONTROL_METHOD == 'AP_PITCH' and ap.engaged: ap.disengage()
            if PITCH_CONTROL_METHOD == 'SAS_STABILITY': control.sas = False
            if conn: conn.close()
            return 0.0

        # --- Simulation Loop ---
        print("  Starting PID control loop...")
        start_time = time.time()
        last_print_time = start_time

        while time.time() - start_time < SIMULATION_DURATION:
            current_heading = flight.heading
            num_updates += 1

            # Calculate heading error and accumulate
            error = TARGET_HEADING - current_heading
            while error > 180: error -= 360
            while error <= -180: error += 360
            total_abs_error += abs(error)

            # Update PID and get control output for Yaw
            yaw_input = pid_controller.update(current_heading)
            control.yaw = yaw_input

            # Optional: Print status periodically
            if time.time() - last_print_time > 5.0:
                 # Added VSI to periodic status print
                 print(f"    t={time.time()-start_time:.1f}s, Head={current_heading:.1f}, Err={error:.1f}, Yaw={yaw_input:.2f}, VSI={flight.vertical_speed:.1f}")
                 last_print_time = time.time()

            time.sleep(CONTROL_UPDATE_INTERVAL)

        # --- Simulation End & Cleanup ---
        print("  Simulation finished.")
        control.yaw = 0 # Reset yaw control first

        # --- Calculate Fitness ---
        if num_updates > 0:
            average_abs_error = total_abs_error / num_updates
            fitness = 1.0 / (average_abs_error + 1e-6) # Higher fitness = lower error
            print(f"  Average Abs Heading Error: {average_abs_error:.4f} -> Fitness: {fitness:.4f}")
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
        if conn:
            try:
                print("  Cleaning up controls...")
                if 'control' in locals():
                    control.yaw = 0
                    # Consider throttling down slightly or fully after test?
                    # control.throttle = 0.5
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
# Increased range slightly based on user input, but 2.5 might still be high for Ki/Kd
gene_space = [
    {'low': 0.0, 'high': 2.5},  # Range for Kp
    {'low': 0.0, 'high': 2.5},  # Range for Ki
    {'low': 0.0, 'high': 2.5}   # Range for Kd
]
num_genes = len(gene_space)

# --- GA Parameters - TUNE THESE! ---
# Reduced parameters based on user input - this will be a very fast but potentially shallow search
num_generations = 10
num_parents_mating = 2
sol_per_pop = 6
parent_selection_type = "sss" # Steady-state selection
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
                       on_generation=generation_countdown_callback # Assign the modified callback
                       )

# --- Run the GA ---
print("\n=== Starting Genetic Algorithm Run ===")
print(f"Target Heading: {TARGET_HEADING} degrees")
print(f"Simulation Duration per Solution: {SIMULATION_DURATION} seconds")
print(f"Pitch Control Method: {PITCH_CONTROL_METHOD}")
if PITCH_CONTROL_METHOD == 'AP_PITCH':
    print(f"Target Initial Pitch: {TARGET_INITIAL_PITCH} degrees")
print("Ensure KSP is running, kRPC server is active,")
print("and the aircraft is in a relatively stable flight state (airborne, decent speed).")
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
        # Check if best_solution_generation attribute exists
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