@with_kw mutable struct lateral_action_space
    go_right::Int = -1          # laneChangeChoice
    go_straight::Int = 0
    go_left::Int = 1
end

# ideas:
# set up proportional lane tracker here
# maybe don't check to see if the lane exists, as we can just penalize for that in the reward function