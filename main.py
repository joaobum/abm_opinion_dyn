import os
from multiprocessing import Pool
import time
import datetime

import numpy as np

from configuration import *
from model import Model
from analysis import Analysis


def instantiate_simulation(config):
    print(f'PID {os.getpid()} running: {config}')
    model = Model(
        config['n_policies'],
        config['init_connections'],
        config['orientations_std'],
        config['emotions_mean'],
        config['emotions_std'],
        config['media_conformities_mean'],
        config['media_conformities_std']
    )
    model.run(N_EPOCHS)
    analysis = Analysis(model_data=model.data)

    if config['save_to_file']:
        analysis.save_to_file()
    if config['plot_analysis']:
        analysis.plot_full_analysis()


if __name__ == "__main__":
    tic = time.perf_counter()
    # Build the config list based on all combinations of each parameter's list
    save_to_file = False
    plot_analysis = True
    config_list = []
    for n_policies in N_POLICIES:
        for init_connections in INIT_CONNECTIONS_PROB:
            for orientations_std in INIT_ORIENTATIONS_STD:
                for emotions_mean in INIT_EMOTIONS_MEAN:
                    for emotions_std in INIT_EMOTIONS_STD:
                        for media_conformities_mean in MEDIA_CONFORMITY_MEAN:
                            for media_conformities_std in MEDIA_CONFORMITY_STD:
                                config_list.append({
                                    'n_policies': n_policies,
                                    'init_connections': init_connections,
                                    'orientations_std': orientations_std,
                                    'emotions_mean': emotions_mean,
                                    'emotions_std': emotions_std,
                                    'media_conformities_mean': media_conformities_mean,
                                    'media_conformities_std': media_conformities_std,
                                    'save_to_file': save_to_file,
                                    'plot_analysis': plot_analysis
                                })
    # Create a process pool without limiting the number of processes
    process_pool = Pool()
    process_pool.map(instantiate_simulation, config_list)

    print(f'{len(config_list)} simulations for {AGENTS_COUNT} agents and {N_EPOCHS} epochs completed in {datetime.timedelta(seconds=(time.perf_counter()-tic))}')


# analysis = Analysis(
#     load_from_path='/Users/joaoreis/Documents/Study/Masters/Final_Project/abm_opinion_dyn/data/11-13-17.30-run-(100ag|500ep|3po|or(σ=0.15)|em(μ=0.2σ=0.1)|me(μ=0.1σ=0.0)|ba=4171.dat')
# analysis.plot_full_analysis()
