using AutomotiveDrivingModels
using Parameters

@with_kw mutable struct laneChanger <: LaneChangeModel{LaneChangeChoice}
    dir::Int = NaN # -1:right, 0:straight, 1:left
    v_des::Float64 # figure out what to do with this
    # don't think I'll need the following parts, since they only have to do with safety
    # threshold_forward::Float64 = 50. # got this from TimLaneChanger
    # threshold_lane_change_gap_forward::Float64 = 10.
    # threshold_lane_change_gap_rear::Float64 = 10.
end

# don't know if I will need set_desired_speed! function

# this function is needed, got most stuff from TimLaneChanger, see if I need to add something in place of model.rec
function observe!(model::laneChanger, scene::Frame{Entity{S, D, I}}, roadway::Roadway, egoid::I) where {E, S, D, I}
    
    vehicle_index = findfirst(egoid, scene)
    veh_ego = scene[vehicle_index]
    v = vel(veh_ego.state)

    curr_lane = get_lane(roadway, veh_ego)
    num_lanes = length(roadway.segments.lanes) 

    # initialize as false, and proceed to check if they are actually true
    left_lane_exists = right_lane_exists = false

    if curr_lane != 1               # corresponds to the rightmost lane
        right_lane_exists = true
    end
    if curr_lane != num_lanes       # corresponds to the leftmost lane
        left_lane_exists = false
    end

    # figure out how to enforce a penalty if we get to this point
    # the problem here is that the ProportionalLaneTracker object likely won't work well if we try to switch to a lane that's not there
    if !right_lane_exists && model.dir == -1    # if we are trying right lane and there is no right lane, 
        model.dir = 0                           # then enforce that we are going straight
    elseif !left_lane_exists && model.dir = 1   # do the same with left lane
        model.dir = 0
    end

    return model
end

Base.rand(rng::AbstractRNG, model::laneChanger) = LaneChangeChoice(model.dir)

