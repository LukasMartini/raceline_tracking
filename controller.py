import numpy as np
from math import atan2, sin, cos, sqrt, tan
from numpy.typing import ArrayLike

from simulator import RaceTrack

def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    """
    Low-level controller: Tracks desired steering angle and velocity.
    desired[0]: Target steering angle (rad)
    desired[1]: Target velocity (m/s)
    """
    assert desired.shape == (2,)

    # state
    current_steer = state[2]
    current_vel = state[3]

    # parameters
    max_steer_vel = parameters[9]
    max_accel = parameters[10]
    min_accel = parameters[8]  

    # steering Control 
    target_steer = desired[0]
    steer_error = target_steer - current_steer
    
    # Gain for steering rate
    K_steer = 10.0 
    v_delta = K_steer * steer_error
    
    #make sure its in the limits 
    v_delta = np.clip(v_delta, -max_steer_vel, max_steer_vel)

    # vlocity Control
    target_vel = desired[1]
    vel_error = target_vel - current_vel
    
    # acceleration gain
    K_vel = 4.0
    a = K_vel * vel_error
    
    #make sure its in the limits 
    a = np.clip(a, min_accel, max_accel)

    return np.array([v_delta, a]).T

def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    """
    High-level controller: Pure Pursuit + Physics-based speed planning.
    Returns: [target_steering_angle, target_velocity]
    """
    # state
    x, y = state[0], state[1]
    yaw = state[4]
    v = state[3]
    
    # parameters
    wheelbase = parameters[0]
    max_steer = parameters[4]
    max_vel = parameters[5]  

    # closest point on track
    car_pos = np.array([x, y])
    distances = np.linalg.norm(racetrack.centerline - car_pos, axis=1)
    closest_idx = np.argmin(distances)
    
    # lookahead, dynamic now
    # Look further ahead as we go faster to ensure stability on straights
    # L = k * v + L_min
    k_lookahead = 0.35 
    min_lookahead = 3.0 
    look_ahead_dist = min_lookahead + k_lookahead * max(v, 0)
    
    # target point
    n = len(racetrack.centerline)
    cumulative_dist = 0
    target_idx = closest_idx
    
    # Search for target point
    for step in range(1, n):
        idx_curr = (closest_idx + step) % n
        idx_prev = (closest_idx + step - 1) % n
        segment_len = np.linalg.norm(racetrack.centerline[idx_curr] - racetrack.centerline[idx_prev])
        cumulative_dist += segment_len
        if cumulative_dist >= look_ahead_dist:
            target_idx = idx_curr
            break
            
    target_point = racetrack.centerline[target_idx]
    
    # steering
    dx = target_point[0] - x
    dy = target_point[1] - y
    
    # Transform target to vehicle frame
    # Rotation matrix R^T * (P_target - P_car)
    local_x = cos(yaw) * dx + sin(yaw) * dy
    local_y = -sin(yaw) * dx + cos(yaw) * dy
    
    # Steering angle delta = atan(k * wheelbase)
    dist_sq = local_x**2 + local_y**2
    
    if dist_sq > 0.01:
        desired_steer = atan2(2 * wheelbase * local_y, dist_sq)
    else:
        desired_steer = 0.0
        
    # physical limits
    desired_steer = np.clip(desired_steer, -max_steer, max_steer)
    
    # Calculate max safe speed for the current required curvature
    # a_lat = v^2 / R  =>  v_max = sqrt(a_lat_max * R)
    # R = wheelbase / tan(steer)
    
    lat_accel_limit = 8.0 # m/s^2
    
    steer_abs = abs(desired_steer)
    if steer_abs > 0.01:
        # Radius of turn
        R = wheelbase / tan(steer_abs)
        v_corner_limit = sqrt(lat_accel_limit * R)
    else:
        v_corner_limit = max_vel

    # slow down if deviating from the path
    cross_track_error = distances[closest_idx]
    if cross_track_error > 1.5:
        # Apply penalty proportional to error
        penalty = np.clip(1.0 - (cross_track_error - 1.5) * 0.2, 0.5, 1.0)
        v_corner_limit *= penalty
        
    # max velocity
    target_v = min(v_corner_limit - 5.0, max_vel - 10.0)
    
    # minimum speed
    target_v = max(target_v, 5.0)
    
    print(f"Lookahead: {look_ahead_dist:.1f}m, Steer: {desired_steer:.2f}rad, V_tgt: {target_v:.1f}m/s")
    
    return np.array([desired_steer, target_v])