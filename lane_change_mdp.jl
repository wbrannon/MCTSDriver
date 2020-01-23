using AutomotiveDrivingModels
using POMDPs
using POMDPModels
using Random
using Parameters
using Statistics

include("mdp_definitions.jl")
include("action_space.jl")
include("lane_change_env.jl")
include("lat_lon_driver.jl")

GOAL_LANE_REWARD = 0.1 #3. #10000.
FINISH_LINE = 1. #100
COLLISION_REWARD = -1. #-5000. #-500.
WAITING_REWARD = -0.00001 #-0.0005 # -1
TIMEOUT_REWARD = -0.8 #0.2 #-0.8 #-50.
BACKWARD_REWARD = -0.8 #-0.4 #-50.
ROAD_END_REWARD =  -0.2 #-0.8 #-50
TOO_SLOW_REWARD = -0.0002 #-0.005 #-5.
OFFROAD_REWARD = -0.0005 #-0.01 #-1. # gets scaled by the amount of distance offroad
HEADING_REWARD = -0.00001 #-0.00005 # gets multiplied by heading
HEADING_TOO_HIGH_REWARD = -0.4
NO_PROGRESS_REWARD = -0.002
PROGRESS_REWARD = 0.2 #5 #0.7 #500.
FINAL_LANE_PROGRESS_REWARD = 1.
FINAL_LANE_RIGHT_REWARD = 1.

EGO_ID = 1


# this should be where the magic happens - where the states, transitions, rewards, etc. are actually called 
@with_kw mutable struct laneChangeMDP <: MDP{Scene, Int64} # figure out what to put within the MDP type
    env::laneChangeEnvironment = laneChangeEnvironment()
    discount_factor::Float64 = 0.95
    terminal_state::Bool = false # this changes after we reach a terminal state (reach goal lane or crash) or we time out (timesteps_allowed reaches zero)
    collision::Bool = false # figure out a collision function
    starting_velocity::Float64 = 10.0
    timestep::Float64 = 0.1
    lat_accel::Float64 = 0.
    long_accel::Float64 = 0.
    model::lat_lon_driver = lat_lon_driver(starting_velocity, timestep)
    # action_space::action_space = action_space()
    driver_models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(EGO_ID => model)
    recommended_speed::Float64 = 10.
    num_features::Int = (1 + env.ncars) * 3
    side_slip::Float64 = 0.
    switching_lanes::Bool = false
    lane_switching_to::Int = 1
    # mdp.prev_action::Int = 1
    prev_direction::Int = 0
    # lane_tracker::LateralDriverModel = ProportionalLaneTracker()
end

# this needs to return the next state and reward
function POMDPs.gen(mdp::laneChangeMDP, s::Scene, a::Int64, rng::AbstractRNG)
    scene = deepcopy(s)
    # define action_map function that maps an integer to an action model
    ego_vehicle = get_by_id(s, EGO_ID)
    ego_lane = get_lane(mdp.env.roadway, ego_vehicle.state).tag.lane
    lane_width = get_lane(mdp.env.roadway, ego_vehicle.state).width
    # if get_lane(mdp.env.roadway, ego_vehicle.state).tag.lane == 1 && a >= 7 && a <= 9
    #     a = 5 # take an unsafe right turn and turn it into a continue straight
    # end
    action, direction = action_map(mdp, a)
    if mdp.env.num_steps <= 10
        a = 5
        direction = 0
    end
    # first, check if we're trying to switch lanes and wheterin the process of switching lanes
    
    
    # if mdp.switching_lanes && mdp.env.current_lane != mdp.lane_switching_to
    #     direction = mdp.prev_direction
    # elseif mdp.switching_lanes && mdp.env.current_lane == mdp.lane_switching_to
    #     mdp.switching_lanes = false
    # end

    # if mdp.env.current_lane == mdp.env.desired_lane
    #     # @show "here"
    #     # @show abs(ego_vehicle.state.posG.y - (mdp.env.desired_lane - 1) * get_lane(mdp.env.roadway, ego_vehicle.state).width)
    #     a = 5
    #     direction = 0
    #     mdp.switching_lanes = false
    #     mdp.prev_direction = 0
    # end
    # if get_lane(mdp.env.roadway, ego_vehicle.state).tag.lane == mdp.env.desired_lane
    # set whether or not we're switching lanes here
    # @show ego_lane
    # @show mdp.lane_switching_to
    if direction == 1 # go left
        # @show "here"
        #XXXXX
        # if mdp.env.current_lane != mdp.env.desired_lane #&& abs(ego_vehicle.state.posG.θ) <= pi/3


            # mdp.switching_lanes = true
            # mdp.prev_direction = 1
            if ego_lane != mdp.env.desired_lane && abs(posf(ego_vehicle.state).t) <= lane_width / 2 # prevent from going offroad to a nonexistent lane here 
                mdp.lane_switching_to = mdp.env.current_lane + 1
                # @show "new lane switch: "
                # @show mdp.lane_switching_to
            end

            # make sure we're not already in the lane
            # if ego_lane != mdp.lane_switching_to && ego_lane < mdp.lane_switching_to
            if ego_lane < mdp.lane_switching_to
                # @show ego_lane
                # @show mdp.lane_switching_to
                t = (ego_lane - mdp.lane_switching_to) * lane_width + posf(ego_vehicle.state).t  #(mdp.lane_switching_to) * get_lane(mdp.env.roadway, ego_vehicle.state).width - ego_vehicle.state.posG.y
                # @show "left"
                # @show ego_lane
                # @show mdp.lane_switching_to
                # @show ego_vehicle.state.posG.y
                # @show posf(ego_vehicle.state).t
                # @show t
            else
                t = posf(ego_vehicle.state).t
                # @show ego_lane
                # @show mdp.lane_switching_to
                # @show "straight, tried to go left"
                # @show t
            end
            # @show "right"
            # @show t
        # else
        #     a = 5
        #     t = posf(ego_vehicle.state).t
        # end


    elseif direction == 0
        t = posf(ego_vehicle.state).t
        # @show ego_lane
        # @show "straight"
        # @show t
    else
        @assert direction == -1 # go right
        #XXXXX
        # if mdp.env.current_lane != 1 #&& abs(ego_vehicle.state.posG.θ) <= pi/3
        #     mdp.switching_lanes = true
        #     mdp.prev_direction = -1
            if ego_lane != 1 && abs(posf(ego_vehicle.state).t) <= lane_width / 2
                mdp.lane_switching_to = mdp.env.current_lane - 1
                # @show "new lane switch: "
                # @show mdp.lane_switching_to
            end
            
            # if ego_lane != mdp.lane_switching_to
            if ego_lane > mdp.lane_switching_to
                # @show ego_lane
                # @show mdp.lane_switching_to
                t = (ego_lane - mdp.lane_switching_to) * lane_width + ego_vehicle.state.posF.t
                # @show "right"
                # @show ego_lane
                # @show mdp.lane_switching_to
                # @show posf(ego_vehicle.state).t
                # @show t
            else
                t = posf(ego_vehicle.state).t
                # @show ego_lane
                # @show mdp.lane_switching_to
                # @show "straight, tried to go right"
                # @show t
                # @show t
            end

            # @show "left"
             #(mdp.lane_switching_to) * get_lane(mdp.env.roadway, ego_vehicle.state).width - ego_vehicle.state.posG.y
        #XXXXX
        # else
        #     t = posf(ego_vehicle.state).t
        #     a = 5
        # end
    end
    dt = velf(ego_vehicle.state).t
    # -t * 3.0 - dt * 2.0
    action = LatLonAccel((-t * 3.0 - dt * 2.0), action.a_lon) # proportional lane tracker
    # @show action.a_lat
    # @show (-t * 3.0 - dt * 2.0)

    # @show ego_lane
    # @show mdp.lane_switching_to
    # @show (-t * 1.0)
    # @show dt
    # @show (-t * 3.0 - dt * 2.0)
    # @show action.a_lat
    # end

    
    # propagate the ego vehicle first, make sure this doesn't cause any issues
    new_ego_state = propagate(ego_vehicle, action, EGO_ID, mdp.env.roadway, mdp.timestep, mdp.lat_accel, mdp.long_accel, mdp.side_slip)
    # scene[vehicle_idx] = Entity(new_ego_state, scene[EGO_ID].def, scene[vehicle_idx].id)
    # ego_vehicle = get_by_id(scene, EGO_ID)
    scene[EGO_ID] = Entity(new_ego_state, scene[EGO_ID].def, scene[EGO_ID].id)
    acts = Vector{LaneFollowingAccel}(undef, length(scene))

    # get the actions of all the other vehicles, this is taken from the get_actions! function in simulate.jl
    for (i, veh) in enumerate(scene)
        # if veh.id != EGO_ID
        if i != EGO_ID
            model = mdp.driver_models[i]
            # observe!(model, scene, mdp.env.roadway, veh.id)
            # line 229, neighbors_features.jl - returns as NeighborLongitudinalResult
            forward_neighbor = get_neighbor_fore_along_lane(scene, i, mdp.env.roadway)
            forward_distance = forward_neighbor.Δs
            forward_idx = forward_neighbor.ind
            if forward_idx != nothing
                forward_vel = scene[forward_idx].state.v
                track_longitudinal!(model, veh.state.v, forward_vel, forward_distance)
            else
                track_longitudinal!(model, scene, mdp.env.roadway, i, forward_neighbor)
            end
            
            acts[i] = rand(rng, model)
            # acts[i] = LaneFollowingAccel(model.a)
            if veh.state.posG.x == mdp.env.road_length
                acts[i] = LaneFollowingAccel(0.)
            end
            # acts[i] = model.a
        end
    end

    # next, propogate the scene for everyone else, this is taken from the tick! function in simulate.jl
    for i in EGO_ID+1:length(scene)
        # vehicle_idx = findfirst(i, scene)
        veh = scene[i]
        new_state = propagate(veh, acts[i], mdp.env.roadway, mdp.timestep)
        scene[i] = Entity(new_state, veh.def, veh.id)
    end
    # update mdp scene
    mdp.env.scene = scene
    return (sp = scene, r=POMDPs.reward(mdp, s, a, scene))
    
end

POMDPs.discount(mdp::laneChangeMDP) = mdp.discount_factor
POMDPs.actions(mdp::laneChangeMDP) = 1:9
POMDPs.n_actions(mdp::laneChangeMDP) = 9
POMDPs.actionindex(mdp::laneChangeMDP, a::Int64) = a
# function POMDPs.n_actions(mdp::laneChangeMDP)
#     action_space_dict = get_action_space_dict(mdp.action_space, mdp.model, env.scene, env.roadway, env.ego_idx)
#     return length(collect(keys(action_space_dict))) # expecting this line to simply return the length of the dictionary
# end

# create an initial scene with all assigned behavioral models - details regarding the HVs are taken care of in lane_change_env.jl
function POMDPs.initialstate(mdp::laneChangeMDP, rng::AbstractRNG)
    # get clean slate for roadway and scene
    mdp.env = laneChangeEnvironment()
    ego_posG = VecSE2(10.,0.,0.)
    curve = mdp.env.roadway[1].lanes[1].curve
    lane = Lane(LaneTag(1, 1), curve)
    ego_vel = mdp.starting_velocity # m/s
    ego_posF = Frenet(ego_posG, lane, mdp.env.roadway)
    mdp.side_slip = 0.
    mdp.switching_lanes = false
    mdp.lane_switching_to = 1
    mdp.prev_direction = 0
    ego_state = VehicleState(ego_posF, mdp.env.roadway, ego_vel) # can change the ego state here, but scene only takes in VehicleState
    ego = Entity(ego_state, VehicleDef(), EGO_ID) 
    push!(mdp.env.scene, ego)
    mdp.env.scene, mdp.env.roadway = create_env(mdp.env)
    mdp.driver_models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(EGO_ID => mdp.model)
    
    # assign behavioral models, for now just go with IDM - the create_env function takes care of assigning velocities randomly
    for i in EGO_ID+1:mdp.env.ncars+1
        mdp.driver_models[i] = IntelligentDriverModel(v_des = mdp.env.scene[i].state.v)
    end
    # not sure if I need to add a burn in period - keep this in mind
    return mdp.env.scene
end

# define reward function, mainly based on whether we reached the goal lane, there was a collision, or we are still going
function POMDPs.reward(mdp::laneChangeMDP, s::Scene, a::Int64, sp::Scene)
    # check if we collide BEFORE we check if we're in the goal lane; otherwise, we might crash to get into the goal lane
    # there is a collision_checker(scene, egoid function)
    # first, check if there is a collision
    
    # next, get the lane that the ego vehicle is in
    ego_veh = get_by_id(sp, EGO_ID) #sp[EGO_ID]
    ego_lane = get_lane(mdp.env.roadway, ego_veh.state).tag.lane
    mdp.env.collision = collision_checker(sp, EGO_ID)
    r = 0.
    if mdp.env.collision
        mdp.env.terminal_state = true
        r += COLLISION_REWARD
    elseif ego_lane == mdp.env.desired_lane && abs(ego_veh.state.posF.ϕ) < 0.1 && abs(ego_veh.state.posF.t) < 0.2 #abs(ego_veh.state.posG.y - (mdp.env.desired_lane ) * get_lane(mdp.env.roadway, ego_veh.state).width) < 1.5  # worked for 0.2
        mdp.env.terminal_state = true
        r += FINISH_LINE
    elseif mdp.env.num_steps >= mdp.env.max_steps # timed out - not sure if this is a good way to do this but let's give it a shot!
        mdp.env.terminal_state = true
        r += TIMEOUT_REWARD
    elseif ego_veh.state.posG.x >= mdp.env.road_length
        mdp.env.terminal_state = true
        r += ROAD_END_REWARD
    # elseif abs(ego_veh.state.posG.θ) > 3.14
    #     mdp.env.terminal_state = true
    #     r += HEADING_TOO_HIGH_REWARD
    end
    # else
        # if ego_lane != mdp.env.desired_lane # make it so it only penalizes for waiting when we're not in the goal lane - trying to get it to straighten out after
        #     r += WAITING_REWARD
        # end
        # r += WAITING_REWARD
        # r += abs(ego_veh.state.posG.θ) * HEADING_REWARD
    # end

    # if ego_lane == mdp.env.desired_lane
    #     old_ego_veh = get_by_id(s, EGO_ID)
    #     # do this so it learns to decrease heading in desired lane
    #     # further, it penalizes if the heading keeps increasing
    #     r += GOAL_LANE_REWARD
    #     r += (abs(old_ego_veh.state.posG.θ) - abs(ego_veh.state.posG.θ)) * FINAL_LANE_PROGRESS_REWARD 
    #     # try to force to turn right into goal lane
    #     # if  a >= 7 && a <= 9 && abs(ego_veh.state.posG.θ) < 2π
    #     #     r += FINAL_LANE_RIGHT_REWARD
    #     # end


    # if ego_lane != mdp.env.desired_lane
    #     desired_lane_diff = mdp.env.desired_lane - ego_lane
    #     r += NO_PROGRESS_REWARD * desired_lane_diff
    # end

    #penalize slowing down too much
    # if ego_veh.state.v < mdp.recommended_speed
    #     r += (mdp.recommended_speed - ego_veh.state.v) * TOO_SLOW_REWARD
    # end

    if ego_lane > mdp.env.current_lane
        r += PROGRESS_REWARD
    end

    # if ego_lane != mdp.env.current_lane


    # if ego_lane != mdp.lane_switching_to && mdp.switching_lanes
    #     mdp.switching_lanes = true

    #XXXXX
    # if mdp.switching_lanes && ego_lane == mdp.lane_switching_to
    #     mdp.switching_lanes = false
    #     mdp.prev_direction = 0
    #     # mdp.env.current_lane = ego_lane # change the assigned lane after the reward is given
    # end

    mdp.env.current_lane = ego_lane
    # if mdp.env.current_lane == mdp.env.desired_lane
    #     print("made it!!!!")
    # end

    # penalize facing backwards
    # if ego_veh.state.posG.θ > π/2 || ego_veh.state.posG.θ < -π/2
    #     r += BACKWARD_REWARD
    # end

    if is_offroad(ego_veh.state, mdp.env) # offroad 
        if abs(get_offroad_distance(ego_veh.state, mdp.env)) > 10.
            # @show "here"
            mdp.env.terminal_state = true
        end
        if ego_lane == mdp.env.desired_lane
            r += OFFROAD_REWARD * get_offroad_distance(ego_veh.state, mdp.env)
        else
            r += 1000 * OFFROAD_REWARD * get_offroad_distance(ego_veh.state, mdp.env)
        end
    end

    mdp.env.num_steps += 1
    return r
end

# the reward function changes the isterminal function, and I believe this should work just fine
POMDPs.isterminal(mdp::laneChangeMDP) = mdp.env.terminal_state

# define a function that returns a vector of features for input into the NN
# for now, define the feature vector as the x and y coordinates of each car, along with their velocities
function POMDPs.convert_s(::Type{V}, s::Scene, mdp::laneChangeMDP) where V<:AbstractArray
    env = mdp.env
    features = ones(mdp.num_features)
    ego_veh = get_by_id(s, EGO_ID)
    features[1] = ego_veh.state.posG.x
    features[2] = ego_veh.state.posG.y
    features[3] = ego_veh.state.v
    veh_idx = EGO_ID + 1
    feature_idx = 1
    while veh_idx <= env.ncars + 1
        veh = get_by_id(s, veh_idx)
        features[3+feature_idx:feature_idx+5] = [veh.state.posG.x, veh.state.posG.y, veh.state.v]
        feature_idx += 3
        veh_idx +=1
    end
    features = normalize_features(features)
    # return convert(Array{Float32}, features)
    return convert(V, features)
end

# not sure if I need to define the transition function - shouldn't need to since the gen and transition function are redundant
# POMDPs.transition()

# define function that takes in an integer (1-9) and returns an action
function action_map(mdp::laneChangeMDP, a::Int64)
    # get safe actions first
    actions = action_space()
    # safe_action_space = get_action_space_dict(actions, mdp.model, mdp.env.scene, mdp.env.roadway, EGO_ID)
    # assign 1-3 to left, 4-6 to straight, and 7-9 to right
    action_string = "slow_straight"
    direction = 0
    if a == 1
        action_string = "slow_left"
        direction = 1
        act = actions.slow_left
    elseif a == 2
        action_string = "normal_left"
        direction = 1
        act = actions.normal_left
    elseif a == 3
        action_string = "speed_left"
        direction = 1
        act = actions.speed_left
    elseif a == 4
        action_string = "slow_straight"
        direction = 0
        act = actions.slow_straight
    elseif a == 5
        action_string = "normal_straight"
        direction = 0
        act = actions.straight
    elseif a == 6
        action_string = "speed_straight"
        direction = 0
        act = actions.speed_straight
    elseif a == 7
        action_string = "slow_right"
        direction = -1
        act = actions.slow_right
    elseif a == 8
        action_string = "normal_right"
        direction = -1
        act = actions.normal_right
    elseif a == 9
        action_string = "speed_right"
        direction = -1
        act = actions.speed_right
    end
    # check if the proposed action is contained within the safe action space - if not, just return straight and it should hopefully work
    # if !safe_action_space[action_string]
    #     direction = 0
    #     act = actions.slow_straight
    # end

    return act, direction
end

# need to normalize feature vector to put in network
function normalize_features(features::Array{Float64})
    # use batch normalization
    vec_sum = sum(features)
    vec_mean = vec_sum / length(features)
    vec_variance = var(features)
    e = 0.001
    features = (features .- vec_mean) / sqrt(vec_variance + e)
    return features
end

function is_offroad(vehicle::VehicleState, env::laneChangeEnvironment)
    y = vehicle.posG.y
    nlanes = env.nlanes
    lane_width = get_lane(env.roadway, vehicle).width
    if y > (nlanes - 0.5) * lane_width || y < - 0.5 * lane_width
        return true
    else
        return false
    end
end

# get the amount that we are offroad so we can scale the penalty accordingly
function get_offroad_distance(vehicle::VehicleState, env::laneChangeEnvironment)
    y = vehicle.posG.y
    nlanes = env.nlanes
    lane_width = get_lane(env.roadway, vehicle).width
    # remember that we start in the middle of the bottom lane
    top_y_bound = (nlanes - 0.5) * lane_width
    bottom_y_bound = (- 0.5 * lane_width)
    if y > top_y_bound
        return y - top_y_bound
    else
        return abs(y - bottom_y_bound)
    end
end

function simulate(mdp::laneChangeMDP, policy::Policy)
    mdp.env.scene = initialstate(mdp, MersenneTwister(0))
    scene_vec = [mdp.env.scene] #Vector{Frame{Entity{VehicleState,VehicleDef,Int64}}}[mdp.env.scene]
    total_reward = 0.
    # @show scene_vec
    # push!(scene_vec, mdp.env.scene)
    # @show scene_vec
    # @show typeof(scene_vec)
    # @show typeof(mdp.env.scene)
    # append!(scene_vec, [mdp.env.scene])
    # scene_vec[1] = mdp.env.scene
    # @show scene_vec
    for i = 1:mdp.env.max_steps
        features = convert_s(Vector{Float32}, mdp.env.scene, mdp)
        # plug features into policy, get a
        val = policy.qnetwork(features)
        a = policy.action_map[argmax(val)]
        # @show get_by_id(mdp.env.scene, EGO_ID).state.v
        new_scene, reward = gen(mdp, mdp.env.scene, a, MersenneTwister(0))
        total_reward += reward
        # @show a
        # @show total_reward
        # @show(get_lane(mdp.env.roadway, new_scene[EGO_ID].state).tag.lane)
        push!(scene_vec, new_scene)
        if mdp.env.terminal_state
            break
        end
    end
    return scene_vec, mdp.env.collision, mdp.env.current_lane == mdp.env.desired_lane
end

