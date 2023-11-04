from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# Connect to vehicle
import argparse
parser = argparse.ArgumentParser(description='commands')
parser.add_argument('--connect')
args = parser.parse_args()

connection_string = args.connect

print(f"Connection to vehicle on {connection_string}")
vehicle = connect(connection_string, wait_ready=True, timeout=300)

# Plan the mission
def arm_and_takeoff(tgt_altitude):
    
    print("Arming motors")
    
    # Sleep until vehicle is armable
    while not vehicle.is_armable:
        time.sleep(1)
    
    # Set to Guided mode and arm the vehicle
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    
    print("Takeoff")
    vehicle.simple_takeoff(tgt_altitude)
    
    # Wait to reach the target altitude
    while True:
        altitude = vehicle.location.global_relative_frame.alt
        
        if altitude >= tgt_altitude - 1:
            print("Altitude Reached")
            break
        
        time.sleep(1)
        
    
# Main program
arm_and_takeoff(10)

# Set the default speed
vehicle.airspeed = 7

# Go to waypoint 1
print("Go to waypoint 1")
wp1 = LocationGlobalRelative(35.9872609, -95.8753037, 10)
vehicle.simple_goto(wp1)


# Stay in this new location for 15 seconds
time.sleep(15)

# Coming back to initial position
print("Coming back")
vehicle.mode = VehicleMode("RTL")  # RTL = Return to Launch

time.sleep(15)

vehicle.close() 
