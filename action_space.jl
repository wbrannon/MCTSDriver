using AutomotiveDrivingModels
using Parameters

# lat_lon_driver is governed by IDM for longitudinal control, ProportionalLaneTracker for lateral control, and MOBIL for lane-change decision making
# goal of this file is to develop the tools needed to return a safe action space 
include("lat_lon_driver.jl")
include("lateral_actions.jl")
include("longitudinal_actions.jl")

NUMBER_DISCRETE_ACTIONS = 9
MAX_SPEED = 50. # m/s 
NORMAL_LAT_ACCEL = 0.5 # m/s^2
# assigning speeds and directions with actions - since the lanes increment to the left, assign positive number with going left
DIR_RIGHT = -1 
DIR_MIDDLE =  0
DIR_LEFT =  1
MIN_LANE_CHANGE_HEADWAY = 10. # m
NUM_DIRECTIONS = 3
TIMESTEP = 0.1
LONG_ACCEL = 1 #* TIMESTEP # m/s^2
LAT_ACCEL = 0.5 #* TIMESTEP # m/s^2
RIGHT_LANE_IDX = 1 # change this if the orientation changes for some reason

# in action space, we should consider whatever actions warrant a difference between speeding up and slowing down.
# for now, only consider speeding up longitudinally and not mess with lateral speeding up
# can maybe just use lat_lon_accel

@with_kw mutable struct action_space
    lat_action_space = lateral_action_space()
    long_action_space = longitudinal_action_space()
end

# assign an integer into every discrete action, should be 9-10 actions total (10 if we include an always safe action)
function get_action(a::Int; action_space = action_space())
    long_accel_space = action_space.long_action_space
    lane_change_space = action_space.lat_action_space
    if a == 1 # slow down, go straight
        return long_accel_space.slow_down, lane_change_space.go_straight
    elseif a == 2 # maintain speed, go straight
        return long_accel_space.maintain_speed, lane_change_space.go_straight
    elseif a == 3 # speed up, go straight
        return long_accel_space.speed_up, lane_change_space.go_straight
    elseif a == 4 # slow down, go left
        return long_accel_space.slow_down, lane_change_space.go_left
    elseif a == 5 # maintain speed, go left
        return long_accel_space.maintain_speed, lane_change_space.go_left
    elseif a == 6 # speed up, go left
        return long_accel_space.speed_up, lane_change_space.go_left
    elseif a == 7 # slow down, go right
        return long_accel_space.slow_down, lane_change_space.go_right
    elseif a == 8 # maintain speed, go right
        return long_accel_space.maintain_speed, lane_change_space.go_right
    elseif a == 9 # speed up, go right
        return long_accel_space.speed_up, lane_change_space.go_right
    end
end

# passes tests
function get_action_test(a::Int)
    return get_action(a)
end

# will actually likely have to use DIR from lane_change_models.jl instead of MAX_LAT_ACCEL
# function action_space()
#     # make LatLonAccel models here, and determine the direction to turn (if applicable) later
#     slow_left = LatLonAccel(LAT_ACCEL, -LONG_ACCEL)     # covers slow_right and slow_left 
#     normal_left = LatLonAccel(LAT_ACCEL, 0.)            # left and right 
#     speed_left = LatLonAccel(LAT_ACCEL, LONG_ACCEL)     # speed_left and speed_right

#     slow_straight = LatLonAccel(0., -LONG_ACCEL) 
#     straight = LatLonAccel(0., 0.)
#     speed_straight = LatLonAccel(0., LONG_ACCEL)

#     slow_right = LatLonAccel(-LAT_ACCEL, -LONG_ACCEL)
#     normal_right = LatLonAccel(-LAT_ACCEL, 0.)
#     speed_right = LatLonAccel(-LAT_ACCEL, LONG_ACCEL)
#     # check if left lane exists and is available
#     # -if there is a vehicle occupying the left lane right beside us, assume not safe
#     # -check velocities of all upcoming and all ahead vehicles, to make sure velocity diff between ego vehicle and HVs will not cause a wreck
#     return action_space(slow_left, normal_left, speed_left, slow_straight, straight, speed_straight, slow_right, normal_right, speed_right)
# end


function get_action_space_dict(actions::action_space, model::lat_lon_driver, scene::Scene, roadway::Roadway, vehicle_idx::Int)
    action_dict = Dict()

    prev_string = "slow_" # start with slow actions and transition to quick actions
    trial_string = "left"
    direction = 1 # corresponds to left turn 
    # action = actions.slow_turn
    for i = 1:NUMBER_DISCRETE_ACTIONS
        # check if potential action is safe
        if is_safe(model, actions, prev_string * trial_string, scene, roadway, vehicle_idx, direction)
            action_dict[prev_string * trial_string] = true
        else
            action_dict[prev_string * trial_string] = false
        end

        # transition speed - goes from slow to normal to speed
        if prev_string == "slow_"
            prev_string = "normal_"
        elseif prev_string == "normal_"
            prev_string = "speed_"
        else
            prev_string = "slow_"
        end

        # transition direction - goes from left to straight to right - it's already left so don't include here
        if i > NUM_DIRECTIONS && i <= 2 * NUM_DIRECTIONS # from i = 4 to i = 6
            trial_string = "straight"
            direction = 0
        elseif i > 2 * NUM_DIRECTIONS   # from i = 7 to 9
            trial_string = "right"
            direction = -1
        end
    end 
    return action_dict
end

# given a potential action, see if it will be safe to partake in 
function is_safe(model::lat_lon_driver, actions::action_space, speed_str::String, scene::Scene, roadway::Roadway, vehicle_idx::Int, direction::Int)
    # action has both a lateral acceleration and a longitudinal acceleration attached
    curr_vel = model.long_model.v_des # CHANGE THIS TO ACTUAL CURRENT VELOCITY
    curr_lane = get_by_id(scene, vehicle_idx).state.posF.roadind.tag.lane
    nlanes = roadway.segments[1].lanes[end].tag.lane
    # check if turning and at what speed to determine the potential acceleration
    if direction != 0 
        if speed_str == "slow_"
            des_accel = actions.slow_right.a_lon

        elseif speed_str == "normal_"
            des_accel = actions.normal_right.a_lon
        else
            des_accel = actions.speed_right.a_lon
        end
    else # if here, we are seeing about going straight
        if speed_str == "slow_"
            des_accel = actions.slow_straight.a_lon
        elseif speed_str == "normal_"
            des_accel = actions.straight.a_lon
        else
            des_accel = actions.speed_straight.a_lon
        end
    end

    distance_traveled = curr_vel * TIMESTEP + des_accel * TIMESTEP^2
    # want to check if the current distance + the distance to be traveled by the neighbor is larger than the distance that we will travel
    if direction == 1 # corresponds to moving to left lane 
        # first check if we're already in the leftmost lane 
        if curr_lane == nlanes return false
        end
        left_front_neighbor = get_neighbor_fore_along_left_lane(scene, vehicle_idx, roadway)
        if left_front_neighbor.ind != nothing
            front_distance_travelled = scene[left_front_neighbor.ind].state.v * TIMESTEP
        else
            front_distance_travelled = 0.
        end

        left_rear_neighbor = get_neighbor_rear_along_left_lane(scene, vehicle_idx, roadway)
        if left_rear_neighbor.ind != nothing
            rear_distance_travelled = scene[left_rear_neighbor.ind].state.v * TIMESTEP
        else
            rear_distance_travelled = 0.
        end
        # use simple strategy for now to check if we are allowed to change lanes 
        if left_front_neighbor.Δs + front_distance_travelled > distance_traveled && left_rear_neighbor.Δs + rear_distance_travelled > distance_traveled
            return true
        else
            return false
        end
    elseif direction == 0
        # get info about front direction
        front_neighbor = get_neighbor_fore_along_lane(scene, vehicle_idx, roadway)
        if front_neighbor.ind != nothing
            front_distance_travelled = scene[front_neighbor.ind].state.v * TIMESTEP
        else
            front_distance_travelled = 0.
        end

        # get info about rear direction
        rear_neighbor = get_neighbor_rear_along_lane(scene, vehicle_idx, roadway)
        if rear_neighbor.ind != nothing
            rear_distance_travelled = scene[rear_neighbor.ind].state.v * TIMESTEP
        else
            rear_distance_travelled = 0.
        end
        if front_neighbor.Δs + front_distance_travelled > distance_traveled && rear_neighbor.Δs + rear_distance_travelled > distance_traveled
            return true
        else
            return false
        end
    else # direction should be -1 (corresponding to right)
        # check if we're in the first lane, in which we cannot move to a right lane
        if curr_lane == RIGHT_LANE_IDX return false
        end
        right_front_neighbor = get_neighbor_fore_along_right_lane(scene, vehicle_idx, roadway)
        if right_front_neighbor.ind != nothing
            front_distance_travelled = scene[right_front_neighbor.ind].state.v * TIMESTEP
        else
            front_distance_travelled = 0.
        end

        right_rear_neighbor = get_neighbor_rear_along_right_lane(scene, vehicle_idx, roadway)
        if right_rear_neighbor.ind != nothing
            rear_distance_travelled = scene[right_rear_neighbor.ind].state.v * TIMESTEP
        else
            rear_distance_travelled = 0.
        end

        if right_front_neighbor.Δs + front_distance_travelled > distance_traveled && right_rear_neighbor.Δs + rear_distance_travelled > distance_traveled
            return true
        else
            return false
        end
    end
end

# get_actions produces a dictionary mapping of strings to booleans, determining whether or not the following 9 actions will produce a crash
# 9 available actions (in this order): [slow left, left, speed left, slow straight, straight, speed straight, slow right, right, speed right]
function get_actions(model::lat_lon_driver, scene::Scene, roadway::Roadway, ego_id::Int)
    max_speed = 50. # use this later 
    max_accel = 2 # m/s^2
    timestep = 0.1 # s
    # vehicle_idx = findfirst(ego_id, scene)
    # try out speed up action first 
    curr_state = get_by_id(scene, ego_id).state
    curr_vel = curr_state.v 
    actions = action_space()
    return get_action_space_dict(actions, model, scene, roadway, ego_id)
end




# NEED TO DO 
# get all actions for ego car - this allows for quicker MCTS solve since we're pruning away unsafe actions right away 
# - this includes combined lateral and longitudinal acceleration
# - find out how to check if this is a safe move