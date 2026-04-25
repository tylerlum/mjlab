# Observation shaping is much more complicated with conditioning
# There are a few possible shapes for the observation
# 1. (N, O): Standard observation from the env
# 2. (N, O + 1): Observation from the env + conditioning index in {0, ..., M - 1}
# 3. (N, O + C): Observation from the env + conditioning
#   * O is the observation dimension
#   * C is the conditioning dimension
#   * M is the number of conditionings
#   * N is the batch size
# If conditioning is off, everything is (1)
# If conditioning is on, everything is (2), except the raw obs from the env is (1) and the input right before forward pass is (3), which is internal to the model
# At a high-level, we are learning conditionings for M agents, so we are learning a conditioning tensor of shape (M, C)
CONDITIONING_IDX_DIM = 1
