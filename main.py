import numpy as np

from configuration import *
from model import Model
from analysis import Analysis
import time


# before = time.time()
for init_connections in INIT_CONNECTIONS_PROB:
    for orientations_std in INIT_ORIENTATIONS_STD:
        for emotions_mean in INIT_EMOTIONS_MEAN:
            for emotions_std in INIT_EMOTIONS_STD:
                for media_conformities_mean in MEDIA_CONFORMITY_MEAN:
                    for media_conformities_std in MEDIA_CONFORMITY_STD: 

                        model = Model(init_connections, 
                                      orientations_std, 
                                      emotions_mean, 
                                      emotions_std, 
                                      media_conformities_mean, 
                                      media_conformities_std)
                        model.run(N_EPOCHS)
                        analysis = Analysis(model_data=model.data)
                        analysis.save_to_file()
                        analysis.plot_full_analysis()
    

    
    
    
# for emotions_mean in [0.5]:
#     for noise_intensity_mean in [0.5]:
#         for policies_negative_ratio in [0.5]:
#             for policies_abs_mean in [0.7]:
#                 for policies_std in [0.1]:
#                     policies_orientations = np.concatenate([np.random.normal(policies_abs_mean, policies_std, int(POLICIES_COUNT * (1 - policies_negative_ratio))), \
#                                                         np.random.normal(-policies_abs_mean, policies_std, int(POLICIES_COUNT * policies_negative_ratio))])
#                     policies_orientations = np.clip(policies_orientations, ORIENTATION_MIN, ORIENTATION_MAX)
                
#                     simulation = Model(policies_orientations, policies_negative_ratio, policies_abs_mean, policies_std, agents_orientation, emotions_mean, emotions_std, noise_intensity_mean, noise_intensity_std)
#                     simulation_info = simulation.run_simulation(n_epochs, interaction_ratio, N_SNAPSHOTS, None, SAVE_RESULTS)

#                     # Now analyse the data on returned snapshots
#                     data_analyser = Analysis(simulation_info=simulation_info)
#                     data_analyser.plot_full_analysis()