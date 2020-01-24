using AutomotiveDrivingModels
using AutoViz
using POMDPs
using POMDPModels
using MCTS
using Random

include("lane_change_mdp.jl")

function simulate_pomdp()
    mdp = laneChangeMDP()
    solver = MCTSSolver(n_iterations=200, depth=80, exploration_constant=10.0)
    planner = solve(solver, mdp)
    mdp.env.scene = initialstate(mdp, MersenneTwister(0))
    scene_vec = [mdp.env.scene]
    r = 0.
    max_steps = 200
    for t = 1:max_steps
        scene = deepcopy(mdp.env.scene) # make copy that won't be messed up
        a = action(planner, mdp.env.scene)
        (mdp.env.scene, reward) = POMDPs.gen(mdp, scene, a, MersenneTwister(0))
        push!(scene_vec, mdp.env.scene)
        mdp.env.num_steps = t
        r += reward
        if isterminal(mdp)
            break
        end
    end
    @show r
    return mdp.env.roadway, scene_vec
end

# roadway, scenes = simulate_pomdp()