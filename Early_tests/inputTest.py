import krpc
import time

try:
    conn = krpc.connect(name='Aircraft Control Example')
    vessel = conn.space_center.active_vessel
    control = vessel.control
    flight_info = vessel.flight(vessel.orbit.body.reference_frame) # Get flight info relative to the body

    print("Taking control...")

    # Initial setup
    control.sas = False # Ensure SAS is off if we want full manual control
    control.rcs = False
    control.throttle = 0.5 # Set throttle to 50%
    print("Throttle set to 50%")

    # Example: Fly straight and level (attempt) - requires constant small adjustments usually
    # For this example, just set a slight pitch up and hold it
    control.pitch = 1.5 # Slight pitch up input
    print("Applying slight pitch up input.")

    # Keep the controls applied for a duration
    duration = 15 # seconds
    print(f"Maintaining controls for {duration} seconds...")
    start_time = time.time()
    while time.time() - start_time < duration:
        altitude = flight_info.mean_altitude
        speed = flight_info.speed
        print(f"Altitude: {altitude:,.0f} m, Speed: {speed:.1f} m/s")

        # You would typically add logic here to adjust pitch/roll/yaw
        # based on sensor readings (altitude, vertical speed, heading etc.)
        # to actually maintain stable flight.

        time.sleep(0.1) # Loop delay

    # Return controls to neutral/off state
    print("Releasing controls...")
    control.pitch = 0.0
    control.throttle = 0.0
    # Maybe turn SAS back on if desired
    control.sas = True
    control.throttle = 1

    print("Control released.")

except krpc.error.RPCError as e:
    print(f"kRPC Error: {e}")
except ConnectionRefusedError:
    print("Connection Refused. Is KSP running with the kRPC server active?")
finally:
    if 'conn' in locals() and conn:
        print("Closing connection.")
        conn.close()