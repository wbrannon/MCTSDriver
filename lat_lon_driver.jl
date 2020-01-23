using AutomotiveDrivingModels

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
    long_model::LaneFollowingDriver = IntelligentDriverModel(v_des=v),
    lat_model::LateralDriverModel = ProportionalLaneTracker(),
    lane_change_model::LaneChangeModel = MOBIL(t)
    )
    return lat_lon_driver(long_model, lat_model, lane_change_model)
end

get_name(::lat_lon_driver) = "lat_lon_driver"

function set_desired_speed!(model::lat_lon_driver, v_des::Float64)
    set_desired_speed!(model.long_model, v_des)
    set_desired_speed!(model.lat_model, v_des)
    return model
end

function AutomotiveDrivingModels.get_lane_offset(a::LaneChangeChoice, scenes::Vector{Frame{E}}, roadway::Roadway, vehicle_index::Int, pastframe::Int=0) where E
    if a.dir == 0 # straight
        return scenes[pastframe][vehicle_index].state.posF.t 
    elseif a.dir == -1 # right turn
        return convert(Float64, get(LANEOFFSETRIGHT, scenes, roadway, vehicle_index, pastframe))
    else
        @assert(a.dir == 1) # left turn
        return convert(Float64, get(LANEOFFSETLEFT, scenes, roadway, vehicle_index, pastframe))
    end
end

# TO DO: 
# possibly resort to QueueRecord instead of Vector of scenes to make everything work
# look into writing track_lateral! function or taking from current implementations
function observe!(driver::lat_lon_driver, scene::Frame{Entity}, roadway::Roadway, egoid::Int)

    # update!(driver.scenes, scene)
    observe!(driver.lane_change_model)

    vehicle_index = findfirst(egoid, scene)
    lane_change_action = rand(driver.lane_change_model)
    lane_offset = get_lane_offset(lane_change_action, driver.scenes, roadway, vehicle_index)
    lateral_speed = convert(Float64, get(VELFT, driver.scenes, roadway, vehicle_index))

    # track_lateral!(driver.lat_model, lane_offset, lateral_speed)
    track_longitudinal!(driver.long_model, scene, roadway, vehicle_index)

    return driver
end