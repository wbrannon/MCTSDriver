using AutomotiveDrivingModels
include("lat_lon_driver.jl")
include("action_space.jl")

# for now, don't account for partial observability
# need to define custom state and custom state transition function

# state is defined by vehicle state (x, y, z)
mutable struct AgentState
    state::VehicleState
    long_accel::Float64
    lat_accel::Float64
end

function AutomotiveDrivingModels.propagate(vehicle::Entity{AgentState,VehicleDef(),Int64}, action::LatLonAccel, roadway::Roadway, timestep::Float64)
    agent = vehicle.state # should pick up the AgentState here
    x = agent.state.posG.x
    y = agent.state.posG.y
    θ = agent.state.posG.θ
    vel = agent.state.v 
    long_accel = agent.long_accel
    lat_accel = agent.lat_accel

    new_long_accel = long_accel + action.a_lon
    new_lat_accel = lat_accel + action.a_lat # want to act in direction of turn

    new_vel = vel + long_accel * timestep 
    new_vel = max(0., new_vel) # prevents reversing

    


    # update longitudinal and lateral acceleration
end