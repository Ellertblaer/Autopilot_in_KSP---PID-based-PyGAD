import krpc
import time
import math
import pygad
import numpy as np # PyGAD uses numpy

# --- Configuration ---
KSP_CONNECTION_NAME = 'GA_PID_Tuner'
TARGET_HEADING = 45  # Target heading in degrees (e.g., 90 for East)
SIMULATION_DURATION = 30  # How many seconds to test each set of PID gains
CONTROL_UPDATE_INTERVAL = 0.1 # How often to update PID and controls (seconds)
INITIAL_THROTTLE = 1.0 # Initial throttle setting for the test flight

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

    def update(self, measured_value):
        current_time = time.time()
        dt = current_time - self._last_time
        if dt == 0:
            return self.output # Avoid division by zero if called too rapidly

        # Calculate heading error (handle 360/0 degree wrap-around)
        error = self.setpoint - measured_value
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360

        # Proportional term
        p_term = self.Kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        self._integral = max(self.integral_limits[0], min(self.integral_limits[1], self._integral))
        i_term = self.Ki * self._integral

        # Derivative term
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

# --- Fitness Function (The core of the GA) ---
# IMPORTANT: This function will be called many times by PyGAD.
# It needs to connect to KSP, run a short simulation, evaluate, and return fitness.
def fitness_func(ga_instance, solution, solution_idx):
    """
    Evaluates a set of PID gains (solution) by running a KSP simulation segment.
    Higher fitness is better.
    """
    Kp, Ki, Kd = solution # Unpack the gains from the GA solution

    print(f"\n--- Evaluating Solution {solution_idx}: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f} ---")

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
        ap = vessel.auto_pilot # Use autopilot for stability assist maybe?
        surface_ref = vessel.orbit.body.reference_frame # Use non-rotating frame for stability
        flight = vessel.flight(surface_ref)

        # --- Simulation Setup ---
        print(f"  Setting up simulation: Target Heading={TARGET_HEADING}, Duration={SIMULATION_DURATION}s")
        pid_controller = PID(Kp, Ki, Kd, TARGET_HEADING)

        # Prepare vessel state:
        # Ensure it's flying somewhat stable before starting PID test
        # You might need manual setup in KSP before running the script
        control.throttle = INITIAL_THROTTLE
        ap.reference_frame = surface_ref
        ap.target_pitch = 30 # Maintain a slight pitch up (adjust as needed)
        ap.target_roll = 0   # Keep wings level
        ap.engage()
        time.sleep(3) # Give autopilot time to stabilize pitch/roll slightly

        print("  Starting PID control loop...")
        start_time = time.time()
        last_print_time = start_time

        # --- Simulation Loop ---
        while time.time() - start_time < SIMULATION_DURATION:
            current_heading = flight.heading
            num_updates += 1

            # Calculate heading error (handle wrap-around)
            error = TARGET_HEADING - current_heading
            if error > 180: error -= 360
            elif error < -180: error += 360
            total_abs_error += abs(error)

            # Update PID and get control output
            yaw_input = pid_controller.update(current_heading) # PID calculates based on error

            # Apply control input (Yaw)
            control.yaw = yaw_input

            # Optional: Print status periodically
            if time.time() - last_print_time > 5.0:
                 print(f"    t={time.time()-start_time:.1f}s, Head={current_heading:.1f}, Err={error:.1f}, Yaw={yaw_input:.2f}")
                 last_print_time = time.time()

            time.sleep(CONTROL_UPDATE_INTERVAL)

        # --- Simulation End & Cleanup ---
        print("  Simulation finished.")
        control.yaw = 0 # Reset yaw control
        ap.disengage() # Disengage autopilot if used
        # control.throttle = 0 # Optional: throttle down

        # --- Calculate Fitness ---
        if num_updates > 0:
            average_abs_error = total_abs_error / num_updates
            # Fitness: Higher is better. Inverse of average error. Add epsilon to avoid division by zero.
            fitness = 1.0 / (average_abs_error + 1e-6)
            print(f"  Average Abs Error: {average_abs_error:.4f} -> Fitness: {fitness:.4f}")
        else:
            print("  No updates occurred, fitness set to 0.")
            fitness = 0.0 # Penalize solutions that failed to run

    except ConnectionRefusedError:
        print("!! KSP Connection Refused. Is KSP running with kRPC server active? Skipping solution.")
        return 0 # Low fitness if connection fails
    except krpc.error.RPCError as e:
        print(f"!! kRPC Error during simulation: {e}. Skipping solution.")
        # Potentially vessel destroyed or invalid state
        return 0 # Low fitness on RPC errors
    except Exception as e:
        print(f"!! Unexpected Error during simulation: {e}. Skipping solution.")
        return 0 # Low fitness on other errors
    finally:
        if conn:
            try:
                # Try to reset controls even if error occurred mid-simulation
                if 'control' in locals():
                    control.yaw = 0
                if 'ap' in locals() and ap.engaged:
                   ap.disengage()
            except krpc.error.RPCError:
                 print("  kRPC error during cleanup (likely disconnected).") # Ignore cleanup errors
            except Exception as e_clean:
                 print(f"  Unexpected error during cleanup: {e_clean}")
            finally:
                 print("  Closing kRPC connection.")
                 conn.close()

    return fitness

# --- Genetic Algorithm Setup ---

# Gene Space: Define the possible range for each gene (Kp, Ki, Kd)
# THESE RANGES ARE CRITICAL - Adjust based on expected values / trial-and-error
gene_space = [
    {'low': 0.0, 'high': 2.5},  # Range for Kp
    {'low': 0.0, 'high': 2.5},  # Range for Ki
    {'low': 0.0, 'high': 2.5}   # Range for Kd
]
num_genes = len(gene_space)

# GA Parameters - TUNE THESE!
num_generations = 10        # Number of iterations
num_parents_mating = 2    # Number of solutions selected as parents
sol_per_pop = 6          # Number of solutions (individuals) in the population
parent_selection_type = "sss" # Steady-state selection
keep_parents = 2          # Number of parents to keep in the next population (-1 means all)
mutation_type = "random"  # Mutation type
mutation_percent_genes = 30 # Percentage of genes to mutate (e.g., 10, 20, 33) - higher for exploration

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
                       # Optional: Save/load progress
                       # save_best_solutions=True,
                       # save_solutions=True,
                       allow_duplicate_genes=False # Avoid wasting evals on identical solutions
                       )

# --- Run the GA ---
print("\n=== Starting Genetic Algorithm Run ===")
print("Ensure KSP is running, kRPC server is active,")
print("and the aircraft is in a relatively stable flight state before proceeding.")
input("Press Enter to start the GA optimization...")

ga_instance.run()

# --- Results ---
print("\n=== Genetic Algorithm Finished ===")
ga_instance.plot_fitness() # Show plot of fitness over generations

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best solution found (Index {solution_idx}):")
print(f"  Kp = {solution[0]:.6f}")
print(f"  Ki = {solution[1]:.6f}")
print(f"  Kd = {solution[2]:.6f}")
print(f"Fitness value of the best solution = {solution_fitness:.6f}")

# You can now use these best Kp, Ki, Kd values in your final PID controller script.
# Optionally, run one last simulation with the best gains found to see it in action.
print("\nOptimization complete. You can manually test the best gains now.")