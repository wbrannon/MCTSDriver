using POMDPModelTools


mutable struct heuristic_policy <: Policy 
    n_actions::Int64
    scene::Scene
end

heuristic_policy(n_actions, scene) = heuristic_policy(n_actions, scene)