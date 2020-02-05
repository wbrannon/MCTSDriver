using AutomotiveDrivingModels
using AutoViz
using POMDPs
using POMDPModels
using POMDPPolicies
using POMDPSimulators
using POMCPOW
using MCTS
using Random

include("lane_change_mdp.jl")

function simulate_pomdp()
    mdp = laneChangeMDP()
    # solver = MCTSSolver(n_iterations=300, depth=80, exploration_constant=10.0)
    solver = DPWSolver(depth=20, exploration_constant=5.0, n_iterations=500, k_state=4., alpha_state=0.125)
    # solver = POMCPOWSolver(criterion=MaxUCB(20.0))
    planner = solve(solver, mdp)
    mdp.env.scene = initialstate(mdp, MersenneTwister(0))
    scene_vec = [mdp.env.scene]
    r = 0.
    planning_timestep = 0.75
    sim_timestep = 0.05
    speedup = round(Int64, planning_timestep / sim_timestep)
    max_steps = 500 # say that steps are 0.05s apart
    # scene = deepcopy(mdp.env.scene)
    a = 1 #action(planner, mdp.env.scene)
    # mdp.env.scene = scene
    headings = []
    for t = 0:max_steps
        scene = deepcopy(mdp.env.scene) # make copy that won't be messed up
        # implement plan only if we're in the first timestep or if we're at a point where we need to plan again
        if t == 0 || mod(t, speedup) == 0 #|| mod(t, speedup) == round(Int64, speedup / 2)
            mdp.timestep = planning_timestep # use planning_timestep only when planning
            a = action(planner, mdp.env.scene)
        end
        mdp.timestep = sim_timestep
        (mdp.env.scene, reward) = POMDPs.gen(mdp, scene, a, MersenneTwister(0))
        push!(scene_vec, mdp.env.scene)
        # make sure that the lane and number of steps gets set back to what they should be
        ego_vehicle = get_by_id(mdp.env.scene, EGO_ID)
        # @show ego_vehicle.state.posF.ϕ
        mdp.env.current_lane = get_lane(mdp.env.roadway, ego_vehicle.state).tag.lane
        mdp.env.num_steps = t + 1
        r += reward
        push!(headings, mdp.env.scene[1].state.posG.θ)
        if isterminal(mdp)
            break
        end
    end
    @show r
    return mdp.env.roadway, planner, scene_vec, headings
end

# roadway, scenes = simulate_pomdp()