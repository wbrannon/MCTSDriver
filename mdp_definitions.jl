using AutomotiveDrivingModels
include("lat_lon_driver.jl")
include("action_space.jl")
include("lane_change_env.jl")

MAX_LONG_ACCEL = 4. # m/s^2
MAX_LAT_ACCEL = 0.5 # m/s^2
MAX_LAT_VEL = 4. # m/s

# for now, don't account for partial observability
# need to define custom state and custom state transition function

# FOR POMDPS.jl, NEED TO DEFINE THE FOLLOWING:
# STATE S: AgentState definition
# STATE TRANSITION S': propagate function
# ACTION SPACE A: action_space.jl - will likely have to figure out how to transfer dictionary to a useful format for POMDPs.jl
# OBSERVATION O: < might be implicitly defined within ADM.jl > - THIS IS ONLY NEEDED FOR POMDP, WE ARE STARTING WITH MDP
# REWARD R: reward_fn
# DISCOUNT FACTOR γ: discount_factor in mdp 
# state is defined by vehicle state (x, y, z) and lateral and longitudinal accelerations
mutable struct AgentState
    state::VehicleState
    # long_accel::Float64
    # lat_accel::Float64
    # side_slip::Float64 # use this to keep track of velocities and theta
end

function AutomotiveDrivingModels.propagate(vehicle::Entity{VehicleState, VehicleDef, Int64}, action::LatLonAccel, egoid::Int, roadway::Roadway, timestep::Float64)
    agent = vehicle.state # should pick up the AgentState here
    x = agent.posG.x
    y = agent.posG.y
    θ = agent.posG.θ
    v = agent.v 
    ϕ = posf(agent).ϕ
    ds = v*cos(ϕ)
    t = posf(agent).t
    dt = v*sin(ϕ)

    timestep² = timestep*timestep
    Δs = ds*timestep + 0.5*action.a_lon*timestep²
    Δt = dt*timestep + 0.5*action.a_lat*timestep²

    ds₂ = ds + action.a_lon*timestep
    dt₂ = dt + action.a_lat*timestep
    # @show dt₂
    # @show ds₂

    # if dt₂ > 1.
    #     dt₂ = 1.
    # elseif dt₂ < -1.
    #     dt₂ = -1.
    # end

    # if ds₂ > 3.
    #     ds₂ = 3.
    # elseif ds₂ < -3.
    #     ds₂ = -3.
    # end


    speed₂ = sqrt(dt₂*dt₂ + ds₂*ds₂)
    v₂ = sqrt(dt₂*dt₂ + ds₂*ds₂) # v is the magnitude of the velocity vector
    ϕ₂ = atan(dt₂, ds₂)

    roadind = move_along(posf(agent).roadind, roadway, Δs)
    footpoint = roadway[roadind]


    posG = VecE2{Float64}(footpoint.pos.x,footpoint.pos.y) + polar(t + Δt, footpoint.pos.θ + π/2)

    posG = VecSE2{Float64}(posG.x, posG.y, footpoint.pos.θ + ϕ₂)


    new_vehicle_state = VehicleState(posG, roadway, v₂)

    return new_vehicle_state
end