using AutomotiveDrivingModels
include("lat_lon_driver.jl")
include("action_space.jl")

MAX_LONG_ACCEL = 4. # m/s^2
MAX_LONG_ACCEL = 2. # m/s^2

# for now, don't account for partial observability
# need to define custom state and custom state transition function

# FOR POMDPS.jl, NEED TO DEFINE THE FOLLOWING:
# STATE S: AgentState definition
# STATE TRANSITION S': propagate function
# ACTION SPACE A: action_space.jl
# OBSERVATION O: < might be implicitly defined within ADM.jl >
# REWARD R: 
# DISCOUNT FACTOR γ:
# state is defined by vehicle state (x, y, z)
mutable struct AgentState
    state::VehicleState
    long_accel::Float64
    lat_accel::Float64
end

function AutomotiveDrivingModels.propagate(vehicle::Entity{AgentState,VehicleDef,Int64}, action::LatLonAccel, roadway::Roadway, timestep::Float64)
    agent = vehicle.state # should pick up the AgentState here
    x = agent.state.posG.x
    y = agent.state.posG.y
    θ = agent.state.posG.θ
    
    vel = agent.state.v 
    long_accel = agent.long_accel           # current longitudinal acceleration
    lat_accel = agent.lat_accel             # current lateral acceleration
    new_long_accel = long_accel + action.a_lon
    # clip to make sure that our acceleration stays within bounds, as is true in reality
    if new_long_accel > MAX_LONG_ACCEL
        new_long_accel = MAX_LONG_ACCEL
    elseif new_long_vel < -MAX_LONG_ACCEL
        new_long_accel = -MAX_LONG_ACCEL
    end

    new_lat_accel = lat_accel + action.a_lat
    if new_lat_accel > MAX_LAT_ACCEL
        new_lat_accel = MAX_LAT_ACCEL
    elseif new_lat_accel < -MAX_LAT_ACCEL
        new_lat_accel = -MAX_LAT_ACCEL
    end

    # for now, getting a lot of this math from lat_lon_accel.jl

    ϕ = agent.state.posF.ϕ                      # lane relative heading
    curr_long_vel = vel * cos(ϕ)                # longitudinal velocity
    curr_long_vel = max(0., curr_long_vel)      # prevents reversing

    curr_lat_vel = vel * sin(ϕ)                 # lateral velocity

    new_long_vel = curr_long_vel + new_long_accel * timestep    # need to update velocities
    new_lat_vel = curr_lat_vel + new_lat_accel * timestep

    new_vel = sqrt(new_long_vel^2 + new_lat_vel^2)  # magnitude of velocity vector

    new_x = x + new_vel * cos(θ) * timestep
    new_y = y + new_vel * sin(θ) * timestep

    # the comment right below might be relevant only if something closer to steering angle actually becomes part of the action space
    # math for new theta - might just have to add the lateral accleration (this is steering rate pretty much) * timestep squared

    new_θ = θ + atan(new_lat_vel, new_long_vel) # keep an eye on this, this is taken from lat_lon_accel.jl

    posG = VecSE2(new_x, new_y, new_θ)
    new_vehicle_state = VehicleState(posG, roadway, new_vel)

    return AgentState(new_vehicle_state, new_long_accel, new_lat_accel)
end