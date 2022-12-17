import numpy as np
import math
import pathlib

############################################################################
# Model parameters
############################################################################

# Model size
N_POLICIES = [3, 7, 15]
AGENTS_COUNT = 50
N_EPOCHS = 1000

# Data capturing
N_SNAPSHOTS = 201
DATA_DIR = f'{pathlib.Path(__file__).parent.resolve()}/data/'

# Agents initialisation
GLOBAL_SOCIAL_SPARSITY = [0.3, 0.5, 0.7]
INIT_CONNECTIONS_PROB = [0.1]
INIT_ORIENTATIONS_MEAN = 0
INIT_ORIENTATIONS_STD = [0.15]
INIT_EMOTIONS_MEAN = [0.35, 0.65]
INIT_EMOTIONS_STD = [0.15]
MEDIA_CONFORMITY_MEAN = [0, 0.3]
MEDIA_CONFORMITY_STD = [0]
INTERACTION_RATIO = [0.1, 0.3, 0.5, 0.7]

# Media parameters
MEDIA_OUTLETS_COUNT = 3
MEDIA_OUTLETS_MEANS = [-0.5, 0, 0.5]
MEDIA_OUTLETS_STD = [0.1, 0.2, 0.1]

# Noise
NOISE_MEAN = 0
NOISE_STD = 0.05

# Candidates parameters
CANDIDATES_COUNT = 2
CANDIDATES_OPINIONS_MEAN = [-0.7, 0.7]
CANDIDATES_OPINIONS_STD = [0.15, 0.15]

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
SEED = 69
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
    
def normpdf(x, var):
    sd = math.sqrt(var)
    return math.exp(-((x/sd)**2)/2) / (sd * math.sqrt(2*math.pi))

create_prob = []
end_prob = []
creations = 0
destructions = 0

def add_create_prob(prob):
    global create_prob
    create_prob.append(prob)
    
def add_end_prob(prob):
    global end_prob
    end_prob.append(prob)

def add_creation():
    global creations
    creations += 1

def add_destruction():
    global destructions
    destructions += 1
    
def print_create_prob():
    global create_prob
    print(np.mean(create_prob))
    
def print_end_prob():
    global end_prob
    print(np.mean(end_prob))
    
def print_creations():
    global creations
    print(creations)
    
def print_destructions():
    global destructions
    print(destructions)