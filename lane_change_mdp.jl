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

# need to reward tune
GOAL_LANE_REWARD = 5000.
FINISH_LINE = 10000.
COLLISION_REWARD = -100000.
WAITING_REWARD = -0.0001
TIMEOUT_REWARD = -20
ROAD_END_REWARD =  -20
TOO_SLOW_REWARD = -1 
OFFROAD_REWARD = -1
HEADING_REWARD = -0.001 # gets multiplied by heading
MAX_HEADING = pi / 3
HEADING_TOO_HIGH_REWARD = 0.
BACKWARD_REWARD = -5000.
PROGRESS_REWARD = 500.

EGO_ID = 1


# this should be where the magic happens - where the states, transitions, rewards, etc. are actually called 
@with_kw mutable struct laneChangeMDP <: MDP{Scene, Int64} # figure out what to put within the MDP type
    env::laneChangeEnvironment = laneChangeEnvironment()
    discount_factor::Float64 = 0.95
    terminal_state::Bool = false # this changes after we reach a terminal state (reach goal lane or crash) or we time out (timesteps_allowed reaches zero)
    collision::Bool = false # figure out a collision function
    starting_velocity::Float64 = 20.0
    timestep::Float64 = 0.75
    model::lat_lon_driver = lat_lon_driver(starting_velocity, timestep)
    driver_models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(EGO_ID => model)
    recommended_speed::Float64 = 18.
end

# this needs to return the next state and reward
function POMDPs.gen(mdp::laneChangeMDP, s::Scene, a::Int, rng::AbstractRNG)
    scene = deepcopy(s)
    # define action_map function that maps an integer to an action model
    ego_vehicle = get_by_id(s, EGO_ID)
    ego_lane = get_lane(mdp.env.roadway, ego_vehicle.state).tag.lane
    lane_width = get_lane(mdp.env.roadway, ego_vehicle.state).width
    r = 0
    accel, direction = action_map(mdp, a)
    # start out for a second going straight to warm up
    # if mdp.env.num_steps <= 2
    #     a = 5
    #     direction = 0
        # assign negative reward for trying to switch to a new lane that doesn't exist - for now, only do this after the warm-up phase.
        # I believe the propagate function will take care of actually switching the action to a non-offroad one
    # if direction == 1 && ego_lane == mdp.env.nlanes || direction == -1 && ego_lane == 1
    #     r += OFFROAD_REWARD
    # end

    # I think that I may have to manually do this next line to increment the acceleration appropriately
    mdp.model.long_model.a += accel
    # place appropriate actions to agent's long_model, lat_model, and lane_change_model here based on whatever a is
    mdp.model.lane_change_model.dir = direction
    observe!(mdp.model, s, mdp.env.roadway, EGO_ID)
    action = LatLonAccel(mdp.model.lat_model.a, mdp.model.long_model.a)
    # propagate the ego vehicle first, make sure this doesn't cause any issues
    new_ego_state = propagate(ego_vehicle, action, EGO_ID, mdp.env.roadway, mdp.timestep)
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
            # @show forward_neighbor
            # @show forward_distance
            # @show forward_idx
            if forward_idx != nothing
                forward_vel = scene[forward_idx].state.v
                AutomotiveDrivingModels.track_longitudinal!(model, veh.state.v, forward_vel, forward_distance)
            else
                AutomotiveDrivingModels.track_longitudinal!(model, veh.state.v, NaN, forward_distance)
            end
            acts[i] = rand(rng, model)
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
    # r += POMDPs.reward(mdp, s, a, scene)
    return (sp = scene, r=r+POMDPs.reward(mdp, s, a, scene))
end

# POMDPs.observations(mdp::laneChangeMDP)

POMDPs.discount(mdp::laneChangeMDP) = mdp.discount_factor
POMDPs.actions(mdp::laneChangeMDP) = collect(1:9)
POMDPs.n_actions(mdp::laneChangeMDP) = 9
POMDPs.actionindex(mdp::laneChangeMDP, a::Int64) = a

# create an initial scene with all assigned behavioral models - details regarding the HVs are taken care of in lane_change_env.jl
function POMDPs.initialstate(mdp::laneChangeMDP, rng::AbstractRNG)
    # get clean slate for roadway and scene
    mdp.env = laneChangeEnvironment()
    ego_posG = VecSE2(40.,0.,0.)
    curve = mdp.env.roadway[1].lanes[1].curve
    lane = Lane(LaneTag(1, 1), curve)
    ego_vel = mdp.starting_velocity # m/s
    ego_posF = Frenet(ego_posG, lane, mdp.env.roadway)
    ego_state = VehicleState(ego_posF, mdp.env.roadway, ego_vel) # can change the ego state here, but scene only takes in VehicleState
    ego = Entity(ego_state, VehicleDef(), EGO_ID) 
    push!(mdp.env.scene, ego)
    mdp.env.scene, mdp.env.roadway = create_env(mdp.env)
    mdp.driver_models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(EGO_ID => mdp.model)
    
    # assign behavioral models, for now just go with IDM - the create_env function takes care of assigning velocities randomly
    for i in EGO_ID+1:mdp.env.ncars+1
        # for now, just go with IDM for other vehicles - switch to lat_lon_driver later, and then human drivers
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
    vehicle_width = ego_veh.def.width
    lane_break_distance = DEFAULT_LANE_WIDTH / 2 - vehicle_width / 2
    ego_lane = get_lane(mdp.env.roadway, ego_veh.state).tag.lane
    mdp.env.collision = collision_checker(sp, EGO_ID)
    r = 0.
    if mdp.env.collision                # penalize if there is a collision, should only be caused by ego vehicle for now
        mdp.terminal_state = true
        r += COLLISION_REWARD
    elseif ego_lane == mdp.env.desired_lane && abs(ego_veh.state.posF.t) < lane_break_distance && abs(ego_veh.state.posF.ϕ) < π/6  # reward for reaching the desired lane
        mdp.terminal_state = true
        r += FINISH_LINE
    elseif ego_lane == mdp.env.desired_lane
        r += GOAL_LANE_REWARD
    elseif mdp.env.num_steps >= mdp.env.max_steps # timed out - not sure if this is a good way to do this but let's give it a shot!
        mdp.terminal_state = true
        r += TIMEOUT_REWARD
    elseif ego_veh.state.posG.x >= mdp.env.road_length  # penalizing for reaching the end of the road without gettting to desired lane
        mdp.terminal_state = true
        r += ROAD_END_REWARD
    else
        mdp.terminal_state = false
        r += WAITING_REWARD
        r += abs(ego_veh.state.posF.ϕ) * HEADING_REWARD
    end


    #penalize slowing down too much
    if ego_veh.state.v < mdp.recommended_speed
        r += (mdp.recommended_speed - ego_veh.state.v) * TOO_SLOW_REWARD
    end

    if abs(ego_veh.state.posF.ϕ) > MAX_HEADING
        r += (abs(ego_veh.state.posF.ϕ) - MAX_HEADING) * HEADING_TOO_HIGH_REWARD
    end

    if abs(ego_veh.state.posF.ϕ) > π/2
        r += BACKWARD_REWARD
    end

    if ego_lane > mdp.env.current_lane && abs(ego_veh.state.posF.ϕ) < pi/2 # don't reward if progress was made by going backwards
        r += PROGRESS_REWARD
        mdp.env.current_lane = ego_lane
    end

    # mdp.env.num_steps += 1
    return r
end

# the reward function changes the isterminal function, and I believe this should work just fine
function POMDPs.isterminal(mdp::laneChangeMDP) 
    ego_veh = get_by_id(mdp.env.scene, EGO_ID) #sp[EGO_ID]
    vehicle_width = ego_veh.def.width
    lane_break_distance = DEFAULT_LANE_WIDTH / 2 - vehicle_width / 2
    ego_lane = get_lane(mdp.env.roadway, ego_veh.state).tag.lane
    mdp.env.collision = collision_checker(mdp.env.scene, EGO_ID)
    # if mdp.env.collision || (ego_lane == mdp.env.desired_lane && abs(ego.state.posF.t) < 0.1) || mdp.env.num_steps >= mdp.env.max_steps || ego_veh.state.posG.x >= mdp.env.road_length
    #     return true
    # else
    #     return false
    # end
    if mdp.env.collision                # penalize if there is a collision, should only be caused by ego vehicle for now
        @show "collision"
        return true
    elseif ego_lane == mdp.env.desired_lane && abs(ego_veh.state.posF.t) < lane_break_distance && abs(ego_veh.state.posF.ϕ) < π/6 # reward for reaching the desired lane
        @show "finish line"
        return true
    elseif mdp.env.num_steps >= mdp.env.max_steps # timed out - not sure if this is a good way to do this but let's give it a shot!
        @show "timeout"
        return true
    elseif ego_veh.state.posG.x >= mdp.env.road_length  # penalizing for reaching the end of the road without gettting to desired lane
        @show "end of road"
        return true
    else
        return false
    end
end

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
    accel, direction = get_action(a)
    return accel, direction
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

function get_observation(mdp::laneChangeMDP)
    # return scene with a little noise added to all the states of the other vehicles - for now, just mess with x-y coods, and not theta or v
    scene = deepcopy(mdp.env.scene)
    dist = Normal()
    veh_idx = EGO_ID + 1
    while veh_idx <= env.ncars + 1
        vehicle = scene[veh_idx]
        vehicle.state.posG.x += rand(dist)
        vehicle.state.posG.y += rand(dist)
        veh_idx += 1
    end
    return scene
end

function simulate(mdp::laneChangeMDP, policy::Policy)
    mdp.env.scene = initialstate(mdp, MersenneTwister(0))
    scene_vec = [mdp.env.scene] #Vector{Frame{Entity{VehicleState,VehicleDef,Int64}}}[mdp.env.scene]
    total_reward = 0.
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
        if mdp.terminal_state
            break
        end
    end
    return scene_vec, mdp.env.collision, mdp.env.current_lane == mdp.env.desired_lane
end

