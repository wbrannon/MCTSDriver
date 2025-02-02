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
    heur_policy = heuristic_policy(n_actions(mdp), mdp.env.scene)
    solver = DPWSolver(depth=10, exploration_constant=5.0, n_iterations=500, k_state=6., alpha_state=0.125, estimate_value=RolloutEstimator(heur_policy))
    # solver = POMCPOWSolver(criterion=MaxUCB(20.0))
    planner = solve(solver, mdp)
    mdp.env.scene = initialstate(mdp, MersenneTwister(0))
    scene_vec = [mdp.env.scene]
    all_actions = []
    r = 0.
    planning_timestep = 1.#mdp.timestep
    sim_timestep = 0.05
    speedup = round(Int64, planning_timestep / sim_timestep)
    max_steps = mdp.env.max_steps # say that steps are 0.05s apart
    # scene = deepcopy(mdp.env.scene)
    a = 0 #action(planner, mdp.env.scene)
    # mdp.env.scene = scene
    headings = []
    for t = 0:max_steps
        scene = deepcopy(mdp.env.scene) # make copy that won't be messed up
        model = deepcopy(mdp.model)
        # implement plan only if we're in the first timestep or if we're at a point where we need to plan again
        if mod(t, speedup) == 0 || mod(t, speedup) == round(Int64, speedup / 2) #|| mod(t, speedup) == round(Int64, speedup / 4) || mod(t, speedup) == round(Int64, 3 * speedup / 4)  
            mdp.timestep = planning_timestep # use planning_timestep only when planning
            a = POMDPs.action(planner, mdp.env.scene, mdp)
            mdp.model = model

        else    # don't change acceleration except every time we call plan - ie, 
            if a == 1 || a == 3 
                a = 2
            elseif a == 4 || a == 6
                a = 5
            elseif a == 7 || a == 9
                a = 8
            end
        end
        push!(all_actions, a)
        # for running gen, use sim_timestep
        mdp.timestep = sim_timestep

        # (mdp.env.scene, reward) = POMDPs.gen(mdp, scene, a, MersenneTwister(0))
        (mdp.env.scene, reward) = propagate_sim(mdp, scene, a, MersenneTwister(0))
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
    return mdp.env.roadway, planner, scene_vec, headings, all_actions
end

# roadway, scenes = simulate_pomdp()