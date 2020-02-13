using AutomotiveDrivingModels
using POMDPs
using POMDPModels
using Random
using Parameters
using Statistics
using LinearAlgebra
using Distributions

include("mdp_definitions.jl")
include("action_space.jl")
include("lane_change_env.jl")
include("lat_lon_driver.jl")
include("heuristic_policy.jl")

# need to reward tune
GOAL_LANE_REWARD = 200.
FINISH_LINE = 1000.
COLLISION_REWARD = -10000.
WAITING_REWARD = -0.001
TIMEOUT_REWARD = -20
ROAD_END_REWARD =  -20
TOO_SLOW_REWARD = -10
TOO_FAST_REWARD = -10
OFFROAD_REWARD = -1
HEADING_REWARD = -0.001 # gets multiplied by heading
MAX_HEADING = pi / 3
HEADING_TOO_HIGH_REWARD = 0. #-0.001
HARD_DECEL_REWARD = -1.
BACKWARD_REWARD = -5000.
PROGRESS_REWARD = 100.
CLOSENESS_REWARD = -10.
NEAR_COLLISION_REWARD = -50.

EGO_ID = 1


# this should be where the magic happens - where the states, transitions, rewards, etc. are actually called 
# @with_kw mutable struct laneChangeMDP <: POMDP{Scene, Int64, Scene} # figure out what to put within the MDP type
@with_kw mutable struct laneChangeMDP <: MDP{Scene, Int64}
    env::laneChangeEnvironment = laneChangeEnvironment()
    discount_factor::Float64 = 0.95
    terminal_state::Bool = false # this changes after we reach a terminal state (reach goal lane or crash) or we time out (timesteps_allowed reaches zero)
    collision::Bool = false # figure out a collision function
    starting_velocity::Float64 = 18.0
    timestep::Float64 = 1.
    model::lat_lon_driver = lat_lon_driver(starting_velocity, timestep)
    driver_models::Dict{Int, DriverModel} = Dict{Int, DriverModel}(EGO_ID => model)
    recommended_low_speed::Float64 = 18.
    recommended_high_speed::Float64 = 23.
end

# this needs to return the next state and reward
function POMDPs.gen(mdp::laneChangeMDP, s::Scene, a::Int, rng::AbstractRNG)
    scene = deepcopy(s) # have to deepcopy here because we end up returning sp (scene) and a reward dependent on s

    # define action_map function that maps an integer to an action model
    ego_vehicle = get_by_id(s, EGO_ID)
    ego_lane = get_lane(mdp.env.roadway, ego_vehicle.state).tag.lane
    lane_width = get_lane(mdp.env.roadway, ego_vehicle.state).width
    accel, direction = action_map(mdp, a)
    r = 0.

    # I think that I may have to manually do this next line to increment the acceleration appropriately
    # if mdp.timestep == 0.75
    mdp.model.long_model.a += accel
    # end

    # place appropriate actions to agent's long_model, lat_model, and lane_change_model here based on whatever a is
    mdp.model.lane_change_model.dir = direction
    observe!(mdp.model, scene, mdp.env.roadway, EGO_ID) # here, we are changing the direction of the car if trying to go offroad, and modifying our lateral acceleration
                                                        # it then calls track_longitudinal and track_lateral

    action = rand(rng, mdp.model) #LatLonAccel(mdp.model.lat_model.a, mdp.model.long_model.a)


    # propagate the ego vehicle first, make sure this doesn't cause any issues
    new_ego_state_real = propagate(ego_vehicle, action, EGO_ID, mdp.env.roadway, mdp.timestep)
    scene[EGO_ID] = Entity(new_ego_state_real, scene[EGO_ID].def, scene[EGO_ID].id)
    # the following vector is meant to hold
    # acts = Vector{LatLonAccel}(undef, length(scene))
    acts_pure_IDM = Vector{LaneFollowingAccel}(undef, length(scene))
    # vel_const = Vector{Float64}(undef, length(scene))

    # get the actions of all the other vehicles, this is taken from the get_actions! function in simulate.jl
    # for (i, veh) in enumerate(scene)
    #     # if veh.id != EGO_ID
    #     if i != EGO_ID
    #         model = mdp.driver_models[i]
    #         # observe!(model, scene, mdp.env.roadway, veh.id)
    #         # line 229, neighbors_features.jl - returns as NeighborLongitudinalResult
            
    #         # if mdp.timestep == 0.75
    #         AutomotiveDrivingModels.observe!(model, scene, mdp.env.roadway, veh.id)
    #         acts[i] = AutomotiveDrivingModels.rand(rng, model)
    #         # acts_pure_IDM[i] = #AutomotiveDrivingModels.rand(rng, model.mlon)
    #         # vel_const[i] = scene[i].state.v
    #         # else
    #         #     # set it up to where only IDM is used while moving along in the simulation
    #         #     forward_neighbor = get_neighbor_fore_along_lane(scene, i, mdp.env.roadway)
    #         #     forward_distance = forward_neighbor.Δs
    #         #     forward_idx = forward_neighbor.ind
    #         #     if forward_idx != nothing
    #         #         forward_vel = scene[forward_idx].state.v
    #         #         AutomotiveDrivingModels.track_longitudinal!(model.mlon, veh.state.v, forward_vel, forward_distance)
    #         #     else
    #         #         AutomotiveDrivingModels.track_longitudinal!(model.mlon, veh.state.v, NaN, forward_distance)
    #         #     end
    #         #     acts[i] = AutomotiveDrivingModels.rand(rng, model.mlon)
    #         # end
    #     end
    # end

    # need to check at finer timestep if a collision is happening (otherwise we might jump across collisions and not detect them)
    # I guess this is a constant acceleration model?
    # if mdp.timestep == 1.#0.75 # only check this if we are in the planning phase - if not, then this isn't needed
    finer_scene = deepcopy(s)
    collision_checker_timestep = 0.1
    num_finer_steps = convert(Int, mdp.timestep / collision_checker_timestep) - 1
    for i=1:num_finer_steps
        ego_vehicle = finer_scene[EGO_ID]
        new_ego_state = propagate(ego_vehicle, action, EGO_ID, mdp.env.roadway, collision_checker_timestep)
        finer_scene[EGO_ID] = Entity(new_ego_state, finer_scene[EGO_ID].def, finer_scene[EGO_ID].id)
        # run 
        for i in EGO_ID+1:length(finer_scene)
            veh = finer_scene[i]
            new_state = const_vel_propagate(veh, mdp.env.roadway, collision_checker_timestep)
            finer_scene[i] = Entity(new_state, veh.def, veh.id)
        end
        # if this if statement is true, then we return the next scene at a smaller timestep than when we plan - this is a little wonky
        if collision_checker(finer_scene, EGO_ID)
            # return(sp=finer_scene, o=POMDPs.observations(mdp), r=POMDPs.reward(mdp, s, a, finer_scene))
            return(sp=finer_scene, r=POMDPs.reward(mdp, s, a, finer_scene))
        end
    end
    # end

    # next, propogate the scene for everyone else, this is taken from the tick! function in simulate.jl
    for i in EGO_ID+1:length(scene)
        # vehicle_idx = findfirst(i, scene)
        veh = scene[i]
        # model = mdp.driver_models[i]
        # acts_pure_IDM[i] = AutomotiveDrivingModels.rand(rng, model.mlon)
        # new_state = propagate(veh, acts_pure_IDM[i], mdp.env.roadway, mdp.timestep)
        new_state = const_vel_propagate(veh, mdp.env.roadway, mdp.timestep)
        # the following if statement is imposing a hack, where if a HV reaches the end of the road it just reappears at the beginning
        if new_state.posG.x == mdp.env.road_length
            posG = VecSE2(0., new_state.posG.y, 0.)
            posF = Frenet(posG, mdp.env.roadway)
            new_state = VehicleState(posF, mdp.env.roadway, veh.state.v)
        end
        scene[i] = Entity(new_state, veh.def, veh.id)
    end

    
    # update mdp scene
    mdp.env.scene = scene
    # return (sp = scene, o=POMDPs.observations(mdp), r=POMDPs.reward(mdp, s, a, scene))
    return (sp = scene, r=POMDPs.reward(mdp, s, a, scene))
end

function POMDPs.observations(mdp::laneChangeMDP)
    # return scene with a little noise added to all the states of the other vehicles - for now, just mess with x-y coods, and not theta or v
    scene = deepcopy(mdp.env.scene)
    dist = Normal(1, 1)
    veh_idx = EGO_ID + 1
    while veh_idx <= mdp.env.ncars + 1
        vehicle = scene[veh_idx]
        x = vehicle.state.posG.x + rand(dist)
        y = vehicle.state.posG.y + rand(dist)
        θ = vehicle.state.posG.θ
        posG = VecSE2(x, y, θ)
        posF = Frenet(posG, mdp.env.roadway)
        noisy_state = VehicleState(posF, mdp.env.roadway, vehicle.state.v)
        scene[veh_idx] = Entity(noisy_state, vehicle.def, vehicle.id)
        veh_idx += 1
    end
    return scene
end

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
    # lat_model = ProportionalLaneTracker()
    for i in EGO_ID+1:mdp.env.ncars+1
        # for now, just go with IDM for other vehicles - switch to lat_lon_driver later, and then human drivers
        # long_model = IntelligentDriverModel(v_des = mdp.env.scene[i].state.v)
        mdp.driver_models[i] = Tim2DDriver(mdp.timestep; mlane=MOBIL(mdp.timestep))
        AutomotiveDrivingModels.set_desired_speed!(mdp.driver_models[i], mdp.env.scene[i].state.v)
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
    ego_veh = get_by_id(sp, EGO_ID)
    vehicle_width = ego_veh.def.width
    vehicle_length = ego_veh.def.length
    lane_break_distance = DEFAULT_LANE_WIDTH / 2 - vehicle_width / 2
    ego_lane = get_lane(mdp.env.roadway, ego_veh.state).tag.lane
    mdp.env.collision = collision_checker(sp, EGO_ID)
    r = 0.
    if mdp.env.collision                # penalize if there is a collision, should only be caused by ego vehicle for now
        mdp.terminal_state = true
        r += COLLISION_REWARD
    elseif ego_lane == mdp.env.desired_lane && abs(ego_veh.state.posF.t) < lane_break_distance && abs(ego_veh.state.posF.ϕ) < π/6  #&& ego_veh.state.v >= mdp.recommended_speed# reward for reaching the desired lane
        mdp.terminal_state = true
        r += FINISH_LINE
    elseif ego_lane == mdp.env.desired_lane && abs(ego_veh.state.posF.ϕ) < MAX_HEADING
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

    # here, we are testing to see if penalizing for near collisions will be helpful
    # nearest_neighbor_distance, nearest_neighbor_idx = get_nearest_neighbor_distance(mdp, sp, EGO_ID)
    # if nearest_neighbor_distance != Inf # should only pass if there are no other cars on the road
    #     # r += CLOSENESS_REWARD / nearest_neighbor_distance
    #     nearest_neighbor = sp[nearest_neighbor_idx]
    #     near_collision_distance_other_lane = vehicle_width * 2.
    #     near_collision_distance_same_lane = vehicle_length * 1.5
    #     if get_lane(mdp.env.roadway, nearest_neighbor.state).tag.lane != ego_lane
    #         if nearest_neighbor_distance ≤ near_collision_distance_other_lane
    #             r += NEAR_COLLISION_REWARD 
    #         end
    #     else
    #         if nearest_neighbor_distance ≤ near_collision_distance_same_lane
    #             r += NEAR_COLLISION_REWARD
    #         end
    #     end
    # end

    # penalize going too slow
    if ego_veh.state.v < mdp.recommended_low_speed
        r += (mdp.recommended_low_speed - ego_veh.state.v) * TOO_SLOW_REWARD
    # also penalize going too fast
    elseif ego_veh.state.v > mdp.recommended_high_speed
        r += (ego_veh.state.v - mdp.recommended_high_speed) * TOO_FAST_REWARD
    end

    # penalize braking too hard
    if mdp.model.long_model.a < -mdp.model.long_model.d_comfort 
        r += abs(mdp.model.long_model.a) - mdp.model.long_model.d_comfort * (HARD_DECEL_REWARD)
    end

    # if abs(ego_veh.state.posF.ϕ) > MAX_HEADING
    #     r += (abs(ego_veh.state.posF.ϕ) - MAX_HEADING) * HEADING_TOO_HIGH_REWARD
    # end

    if abs(ego_veh.state.posF.ϕ) > π/2
        r += BACKWARD_REWARD
    end

    if ego_lane > mdp.env.current_lane && abs(ego_veh.state.posF.t) < lane_break_distance && abs(ego_veh.state.posF.ϕ) < MAX_HEADING # don't reward if progress was made by going backwards
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
    elseif ego_lane == mdp.env.desired_lane && abs(ego_veh.state.posF.t) < lane_break_distance && abs(ego_veh.state.posF.ϕ) < π/6 #&& ego_veh.state.v >= mdp.recommended_speed# reward for reaching the desired lane
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

# for policy, use something very close to MOBIL 
function POMDPs.solve(solver::DPWSolver, mdp::laneChangeMDP)
    policy = heuristic_policy(n_actions(mdp), mdp.env.scene)
    
end

function POMDPs.action(policy::heuristic_policy, scene::Scene, mdp::laneChangeMDP)
    ego_veh = get_by_id(mdp.env.scene, EGO_ID) #sp[EGO_ID]
    ego_lane = get_lane(mdp.env.roadway, ego_veh.state).tag.lane
    ego_vel = ego_veh.state.v
    vehicle_length = ego_veh.def.length
    # for now, just assume that we're going to the leftmost lane
    scene = mdp.env.scene 

    fore_M = get_neighbor_fore_along_lane(scene, EGO_ID, mdp.env.roadway, VehicleTargetPointFront(), VehicleTargetPointRear(), VehicleTargetPointFront())
    rear_M = get_neighbor_rear_along_lane(scene, EGO_ID, mdp.env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear())

    if ego_lane < mdp.env.desired_lane
        rear_L = get_neighbor_rear_along_left_lane(scene, EGO_ID, mdp.env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear())
        fore_L = get_neighbor_fore_along_left_lane(scene, EGO_ID, mdp.env.roadway, VehicleTargetPointFront(), VehicleTargetPointFront(), VehicleTargetPointRear())
        # candidate position after lane change is over
        footpoint = get_footpoint(ego_veh)
        lane = get_lane(mdp.env.roadway, ego_veh) 
        lane_L = mdp.env.roadway[LaneTag(lane.tag.segment, lane.tag.lane + 1)]
        roadproj = proj(footpoint, lane_L, mdp.env.roadway)
        frenet_L = Frenet(RoadIndex(roadproj), mdp.env.roadway)
        egostate_L = VehicleState(frenet_L, mdp.env.roadway, vel(ego_veh.state))

        Δaccel_n = 0.0
        passes_safety_criterion_rear = true
        passes_safety_criterion_fore = true
        gap_fore = Inf
        gap_rear = Inf
        if rear_L.ind != nothing
            id = scene[rear_L.ind].id
            # accel_n_orig = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a

            # update ego state in scene
            # scene[vehicle_index] = Entity(veh_ego, egostate_L)
            # accel_n_test = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a

            body = inertial2body(get_rear(scene[EGO_ID]), get_front(scene[rear_L.ind])) # project ego to be relative to target
            s_gap_rear = body.x
            gap_rear = s_gap_rear
            
            # restore ego state
            # scene[vehicle_index] = veh_ego
            passes_safety_criterion_rear = s_gap_rear > 3. #accel_n_test ≥ -model.safe_decel && s_gap ≥ 0
            # total_gap = s_gap_rear
            # Δaccel_n = accel_n_test - accel_n_orig
        end

        if fore_L.ind != nothing
            id = scene[fore_L.ind].id
            body = inertial2body(get_rear(scene[EGO_ID]), get_front(scene[fore_L.ind])) # project ego to be relative to target
            s_gap_fore = body.x
            gap_fore = s_gap_fore
            passes_safety_criterion = s_gap_fore > 3.
        end

        if passes_safety_criterion_rear && passes_safety_criterion_fore
            if fore_L.ind == nothing
                return 6
            end
            if gap_fore > 5. && scene[fore_L.ind].state.v - ego_vel > -2.
                if fore_M.ind != nothing
                    if !(scene[fore_M.ind].state.posG.x - scene[EGO_ID].state.posG.x > 3.)
                        return 5
                    else
                        return 6
                    end
                end
            else
                if scene[fore_L.ind].state.v - ego_vel < 0.
                    return 4
                else
                    return 5
                end
            end     
        end

    end
    if fore_M.ind != nothing #scene[fore_M.ind]
        gap = scene[fore_M.ind].state.posG.x - scene[EGO_ID].state.posG.x
        if gap < 3.
            return 1
        end
    end

    return 2
end

# define function that takes in an integer (1-9) and returns an action
function action_map(mdp::laneChangeMDP, a::Int64)
    # get safe actions first
    accel, direction = get_action(a)
    return accel, direction
end

function get_nearest_neighbor_distance(mdp::laneChangeMDP, scene::Scene, ego_id::Int)
    ncars = mdp.env.ncars
    if ncars == 0
        return Inf
    end
    starting_idx = ego_id + 1
    ego_position = scene[ego_id].state.posG
    ego_coods = [ego_position.x, ego_position.y]
    distances = []
    for i=starting_idx:ncars+1
        car_position = scene[i].state.posG
        car_coods = [car_position.x, car_position.y]
        push!(distances, norm(car_coods - ego_coods))
    end
    nearest_neighbor_distance = minimum(distances)
    nearest_neighbor_idx = argmin(distances)
    return nearest_neighbor_distance, nearest_neighbor_idx
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


function propagate_sim(mdp::laneChangeMDP, s::Scene, a::Int, rng::AbstractRNG)
    scene = deepcopy(s) # have to deepcopy here because we end up returning sp (scene) and a reward dependent on s

    # define action_map function that maps an integer to an action model
    ego_vehicle = get_by_id(s, EGO_ID)
    ego_lane = get_lane(mdp.env.roadway, ego_vehicle.state).tag.lane
    lane_width = get_lane(mdp.env.roadway, ego_vehicle.state).width
    accel, direction = action_map(mdp, a)
    r = 0.

    # I think that I may have to manually do this next line to increment the acceleration appropriately
    # if mdp.timestep == 0.75
    mdp.model.long_model.a += accel
    # end

    # place appropriate actions to agent's long_model, lat_model, and lane_change_model here based on whatever a is
    mdp.model.lane_change_model.dir = direction
    observe!(mdp.model, scene, mdp.env.roadway, EGO_ID) # here, we are changing the direction of the car if trying to go offroad, and modifying our lateral acceleration
                                                        # it then calls track_longitudinal and track_lateral

    action = LatLonAccel(mdp.model.lat_model.a, mdp.model.long_model.a)
    new_ego_state_real = propagate(ego_vehicle, action, EGO_ID, mdp.env.roadway, mdp.timestep)
    scene[EGO_ID] = Entity(new_ego_state_real, scene[EGO_ID].def, scene[EGO_ID].id)
    # the following vector is meant to hold

    
    acts = Vector{LatLonAccel}(undef, length(scene))
    # acts_pure_IDM = Vector{LaneFollowingAccel}(undef, length(scene))
    for (i, veh) in enumerate(scene)
    # if veh.id != EGO_ID
        if i != EGO_ID
            model = mdp.driver_models[i]
            veh = scene[i]

            AutomotiveDrivingModels.observe!(model, scene, mdp.env.roadway, veh.id)
            acts[i] = AutomotiveDrivingModels.rand(rng, model)

            new_state = propagate(veh, acts[i], mdp.env.roadway, mdp.timestep)
            if new_state.posG.x == mdp.env.road_length
                posG = VecSE2(0., new_state.posG.y, 0.)
                posF = Frenet(posG, mdp.env.roadway)
                new_state = VehicleState(posF, mdp.env.roadway, veh.state.v)
            end
            scene[i] = Entity(new_state, veh.def, veh.id)
        end
    end

    mdp.env.scene = scene
    return (scene, POMDPs.reward(mdp, s, a, scene))
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

