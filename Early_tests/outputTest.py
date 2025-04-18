#SETUP
import time
import krpc
import math
conn = krpc.connect(name='Sub-orbital flight')
vessel = conn.space_center.active_vessel
print(vessel.name)



#Hér byrjar fyrir alvöru


#Gefur fullt af upplýsingum
    #Skilgreinir allt
conn = krpc.connect(name='Pitch/Heading/Roll')


vessel = conn.space_center.active_vessel

def cross_product(u, v):
    return (u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0])


def dot_product(u, v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def magnitude(v):
    return math.sqrt(dot_product(v, v))


def angle_between_vectors(u, v):
    """ Compute the angle between vector u and v """
    dp = dot_product(u, v)
    if dp == 0:
        return 0
    um = magnitude(u)
    vm = magnitude(v)
    return math.acos(dp / (um*vm)) * (180. / math.pi)


    #Inf. loop sem les á 1Hz.
while True:

    vessel_direction = vessel.direction(vessel.surface_reference_frame)

    # Get the direction of the vessel in the horizon plane
    horizon_direction = (0, vessel_direction[1], vessel_direction[2])

    # Compute the pitch - the angle between the vessels direction and
    # the direction in the horizon plane
    pitch = angle_between_vectors(vessel_direction, horizon_direction)
    if vessel_direction[0] < 0:
        pitch = -pitch

    # Compute the heading - the angle between north and
    # the direction in the horizon plane
    north = (0, 1, 0)
    heading = angle_between_vectors(north, horizon_direction)
    if horizon_direction[2] < 0:
        heading = 360 - heading

    # Compute the roll
    # Compute the plane running through the vessels direction
    # and the upwards direction
    up = (1, 0, 0)
    plane_normal = cross_product(vessel_direction, up)
    # Compute the upwards direction of the vessel
    vessel_up = conn.space_center.transform_direction(
        (0, 0, -1), vessel.reference_frame, vessel.surface_reference_frame)
    # Compute the angle between the upwards direction of
    # the vessel and the plane normal
    roll = angle_between_vectors(vessel_up, plane_normal)
    # Adjust so that the angle is between -180 and 180 and
    # rolling right is +ve and left is -ve
    if vessel_up[0] > 0:
        roll *= -1
    elif roll < 0:
        roll += 180
    else:
        roll -= 180

    print('pitch = % 5.1f, heading = % 5.1f, roll = % 5.1f' %
          (pitch, heading, roll))

    time.sleep(1)