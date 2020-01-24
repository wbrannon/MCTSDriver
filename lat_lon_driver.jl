using AutomotiveDrivingModels
include("accel_driver.jl")
include("lateral_actions.jl")
include("lane_change_driver.jl")


# driver behavioral model: separate longitudinal versus lateral model
# need to finish observe! function and implement rand function (for action sampling)
# for action model, will possibly have to write a propagate function specific to this purpose (but maybe not since using IDM and MOBIL)
# use Frame{E} instead of Scene and Vector{Frame{E}} instead of QueueRecord
mutable struct lat_lon_driver <:DriverModel{LatLonAccel}
    # scenes::Vector{Frame{E}} where E
    long_model::LaneFollowingDriver
    lat_model::LateralDriverModel
    lane_change_model::LaneChangeModel
end

function lat_lon_driver(
    v::Float64, t::Float64; 
    long_model::LaneFollowingDriver = acceleratingDriver(),
    lat_model::LateralDriverModel = ProportionalLaneTracker(),
    lane_change_model::LaneChangeModel = laneChanger()
    )
    return lat_lon_driver(long_model, lat_model, lane_change_model)
end

get_name(::lat_lon_driver) = "lat_lon_driver"

function set_desired_speed!(model::lat_lon_driver, v_des::Float64)
    set_desired_speed!(model.long_model, v_des)
    set_desired_speed!(model.lat_model, v_des)
    return model
end

function AutomotiveDrivingModels.get_lane_offset(a::LaneChangeChoice, scene::Scene, roadway::Roadway, vehicle_index::Int) where E
    veh_ego = scene[vehicle_index]
    t = posf(veh_ego.state).t
    lane = get_lane(roadway, veh_ego)
    if a.dir == 0 # straight
        lane_offset =  scene[vehicle_index].state.posF.t 
    elseif a.dir == -1 # right turn
        if n_lanes_right(lane, roadway) > 0
            lane_right = roadway[LaneTag(lane.tag.segment, lane.tag.lane - 1)]
            lane_offset = t + lane.width/2 + lane_right.width/2
            lane_offset = FeatureValue(lane_offset)
        else
            lane_offset = FeatureValue(NaN, FeatureState.MISSING)
        end
    else
        @assert(a.dir == 1) # left turn
        if n_lanes_left(lane, roadway) > 0
            lane_left = roadway[LaneTag(lane.tag.segment, lane.tag.lane + 1)]
            lane_offset = t - lane.width/2 - lane_left.width/2
            lane_offset = FeatureValue(lane_offset)
        else
            lane_offset = FeatureValue(NaN, FeatureState.MISSING)
        end
        # return convert(Float64, get(LANEOFFSETLEFT, scene, roadway, vehicle_index, 0))
    end
    return convert(Float64, lane_offset)
end

# TO DO: 
# possibly resort to QueueRecord instead of Vector of scenes to make everything work
# look into writing track_lateral! function or taking from current implementations
function observe!(driver_model::lat_lon_driver, scene::Frame{Entity{S,D,I}}, roadway::Roadway, egoid::Int) where {S, D, I}

    # update!(driver.scenes, scene)
    observe!(driver_model.lane_change_model, scene, roadway, egoid)

    ego_vehicle = scene[egoid]
    vehicle_index = findfirst(egoid, scene)
    lane_change_action = rand(driver_model.lane_change_model)
    lane_offset = get_lane_offset(lane_change_action, scene, roadway, vehicle_index)
    lateral_speed = velf(ego_vehicle.state).t #convert(Float64, get(VELFT, scene, roadway, vehicle_index))

    AutomotiveDrivingModels.track_lateral!(driver_model.lat_model, lane_offset, lateral_speed)
    track_longitudinal!(driver_model.long_model)

    return driver_model
end

Base.rand(rng::AbstractRNG, driver::lat_lon_driver) = LatLonAccel(rand(rng, driver.mlat), rand(rng, driver.mlon).a)