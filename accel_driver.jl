using AutomotiveDrivingModels


MAX_ACCEL = 3.
COMFORTABLE_DECEL = 2.
MAX_DECEL = 6.

# make lane change driver almost exactly like Tim's lane changer, except don't check for safety when trying to change

# come up with better name here
# the point of this driver model is largely to mimic the functionality of IDM but without the limitations that IDM poses
# - we want to be able to accelerate whenever we want
# no need to define the propogate function here, as it is already defined in lane_following_accel.jl
@with_kw mutable struct acceleratingDriver <: LaneFollowingDriver
    a::Float64 = NaN # longitudinal acceleration passed here
    σ::Float64 = NaN # optional stdev on top of the model, set to zero or NaN for deterministic behavior
    a_max::Float64 = MAX_ACCEL
    d_comfort::Float64 = COMFORTABLE_DECEL # positive
    d_max::Float64 = MAX_DECEL # positive
end

# don't need track_longitudinal! function, as that is based on headway - to make it work nicely with the way things are currently set out, just define it here
track_longitudinal!(model::acceleratingDriver, v_ego::Float64, v_oth::Float64, headway::Float64) = model

reset_hidden_state!(model::acceleratingDriver) = model

function Base.rand(rng::AbstractRNG, model::acceleratingDriver)
    @assert model.a != NaN # can get rid of this line once everything is working
    if isnan(model.σ) || model.σ ≤ 0.0
        return LaneFollowingAccel(model.a)
    else
        LaneFollowingAccel(rand(rng, Normal(model.a, model.σ)))
    end
end

# don't think I need to define pdf or logpdf function