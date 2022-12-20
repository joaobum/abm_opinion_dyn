import numpy as np
import math
import pathlib

############################################################################
# Model parameters
############################################################################

# Model size
N_POLICIES = [3, 7, 15]
AGENTS_COUNT = 120
N_EPOCHS = 600

# Data capturing
N_SNAPSHOTS = 201
DATA_DIR = f'{pathlib.Path(__file__).parent.resolve()}/data/'

# Agents initialisation
GLOBAL_SOCIAL_SPARSITY = [0.2, 0.4, 0.6, 0.8]
INIT_CONNECTIONS_PROB = [0.15]
INIT_ORIENTATIONS_MEAN = 0
INIT_ORIENTATIONS_STD = [0.15]
INIT_EMOTIONS_MEAN = [0.3, 0.5, 0.7]
INIT_EMOTIONS_STD = [0.15, 0.25]
MEDIA_CONFORMITY_MEAN = [0, 0.3]
MEDIA_CONFORMITY_STD = [0, 0.15]
INTERACTION_RATIO = [0.2, 0.4, 0.6]

# Media parameters
MEDIA_OUTLETS_MEANS = [-0.85, -0.5, 0.4, 0.65]
MEDIA_OUTLETS_STD = [0.1, 0.15, 0.2, 0.15]
MEDIA_OUTLETS_COUNT = len(MEDIA_OUTLETS_MEANS)

# Noise
NOISE_MEAN = 0
NOISE_STD = 0.05

# Candidates parameters
CANDIDATES_OPINIONS_MEAN = [-0.85, -0.35, 0.4, 0.65]
CANDIDATES_OPINIONS_STD = [0.1, 0.15, 0.2, 0.15]
CANDIDATES_COUNT = len(CANDIDATES_OPINIONS_MEAN)

# Boundaries
OPINION_MAX = 1
OPINION_MIN = -1
ORIENTATION_MAX = 1
ORIENTATION_MIN = -1
EMOTION_MIN = 0.05
EMOTION_MAX = 0.95
NOISE_MIN = 0
NOISE_MAX = 1
MEDIA_CONFORMITY_MIN = 0
MEDIA_CONFORMITY_MAX = 1

# Random seed for consistency
SEED = 1
np.random.seed(SEED)

# Tracing
V_AGENT = 1
V_MODEL = 2
V_ANALYSIS = 4
V_MAIN = 8
# Changing the multiplier to 0 disables to 1 enables
# each logging zone
VERBOSITY = V_AGENT     * 0 + \
            V_MODEL     * 0 + \
            V_ANALYSIS  * 0 + \
            V_MAIN      * 0

############################################################################
# Helper functions
############################################################################

def print_array(array: np.ndarray):
    print(array.round(2))