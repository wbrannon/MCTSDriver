using BSON: @load
using DeepQLearning
using Flux
using Random

include("lane_change_mdp.jl")

policy_fname = "policy1.bson"
@load policy_fname policy 
ncars = 9
input = rand(Float32, ncars) * 100

mdp = laneChangeMDP()
scene_vec = simulate(mdp, policy)