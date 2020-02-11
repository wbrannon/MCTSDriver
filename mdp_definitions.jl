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
    # if sign(dt) == 1
    #     dt = min(dt, 5.)
    # else
    #     dt = max(dt, -5.)
    # end

    max_heading_change_rate = pi/6  # assume that in one second, our heading can be changed by 30 degrees at most

    timestep² = timestep*timestep
    Δs = ds*timestep + 0.5*action.a_lon*timestep²
    Δt = dt*timestep + 0.5*action.a_lat*timestep²

    ds₂ = ds + action.a_lon*timestep
    ds₂ = max(ds₂, 0.)
    dt₂ = dt + action.a_lat*timestep
    # should maybe cap out the lateral velocity

    if ds₂ == 0. # if the car isn't moving longitudinally, then our state stays the same
        return agent
    # elseif ds₂ < 1.
    end

    v₂ = sqrt(dt₂*dt₂ + ds₂*ds₂) # v is the magnitude of the velocity vector
    # length = Base.length(vehicle.def)
    # radius = length / 2
    # omega = dt₂ / radius

    ϕ₂ = atan(Δt, Δs) # angular velocity if using dt₂ and ds₂
    # if abs(ϕ₂) > max_heading_change_rate 
    #     ϕ₂ = sign(ϕ₂) * max_heading_change_rate
    # end

    roadind = move_along(posf(agent).roadind, roadway, Δs)
    footpoint = roadway[roadind]


    posG = VecE2{Float64}(footpoint.pos.x,footpoint.pos.y) + polar(t + Δt, footpoint.pos.θ + π/2)

    new_θ = footpoint.pos.θ + ϕ₂ #* timestep
    # do some math to make sure our lane change rate is appropriate
    # max_lane_change_rate = 2/3 * DEFAULT_LANE_WIDTH * t # this places us at 0.67 lanes/s
    # curr_lane_change_rate = v₂ * sin(new_θ) / DEFAULT_LANE_WIDTH
    # if abs(curr_lane_change_rate) > max_lane_change_rate
    #     slowing_factor = abs(max_lane_change_rate / curr_lane_change_rate)
    #     v₂ *= slowing_factor
    # end
    # if abs(new_θ - θ) > max_heading_change
    #     new_θ = footpoint.pos.θ + sign(ϕ₂) * max_heading_change
    # end
    posG = VecSE2{Float64}(posG.x, posG.y, new_θ) # also don't want to allow y to change a ton


    new_vehicle_state = VehicleState(posG, roadway, v₂)

    return new_vehicle_state
end