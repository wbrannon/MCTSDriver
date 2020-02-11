SPEED_UP_ACCEL = 1.
SLOW_DOWN_ACCEL = -1.

MAX_ACCEL = 3.
MAX_DECEL = -6.

@with_kw mutable struct longitudinal_action_space
    speed_up::Float64 = 2.      # m/s^2
    maintain_speed::Float64 = 0.
    slow_down::Float64 = -1.
end

# ideas
# to effectively get this going, should maybe develop a new longitudinal behavioral model that doesn't depend on the headway like IDM does
# - if I do this, will likely use most of the IDM stuff but not the acceleration stuff
# - will still include the headway stuff and all that jazz to check for collisions and peanlize accordingly