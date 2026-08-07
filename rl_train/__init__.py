# Imported for its side effect: rl_train.envs registers the myoAssistLeg* gym env ids
# at import time, so `import rl_train` has to be enough to make gym.make find them.
import rl_train.envs  # noqa: F401
