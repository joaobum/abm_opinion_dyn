import os
from multiprocessing import Pool
import time
import datetime

import numpy as np

from configuration import *
from model import Model
from analysis import ModelAnalysis, StatisticalAnalysis

# Running mode constants
MODEL_TEST = 0
PARAMETER_SWEEP = 1
STATISTICAL_ANALYSIS = 2

def instantiate_simulation(config):
    print(f'PID {os.getpid()} running -> {config}')
    model = Model(
        config['n_epochs'],
        config['n_policies'],
        config['social_sparsity'],
        config['interaction_ratio'],
        config['init_connections'],
        config['orientations_std'],
        config['emotions_mean'],
        config['emotions_std'],
        config['media_conformities_mean'],
        config['media_conformities_std']
    )
    model.run()
    analysis = ModelAnalysis(model_data=model.data)

    if config['save_to_file']:
        analysis.save_to_file()
    if config['plot_analysis']:
        analysis.plot_model_analysis()
        
    return model


if __name__ == "__main__":
    # Running more flags
    running_mode = STATISTICAL_ANALYSIS
    
    if running_mode == MODEL_TEST:
        config = {
            'n_epochs': 600,
            'n_policies': 3,
            'social_sparsity': 0.65,
            'interaction_ratio': 0.3,
            'init_connections': 0.15,
            'orientations_std': 0.15,
            'emotions_mean': 0.5,
            'emotions_std': 0.2,
            'media_conformities_mean': 0.15,
            'media_conformities_std': 0.05,
            'save_to_file': True,
            'plot_analysis': True
        }
        
        instantiate_simulation(config)

    elif running_mode == PARAMETER_SWEEP:
        # Build the config list based on all combinations of each parameter's list
        save_to_file = True
        plot_analysis = False
        config_list = []

        for n_policies in N_POLICIES:
            for social_sparsity in GLOBAL_SOCIAL_SPARSITY:
                for interaction_ratio in INTERACTION_RATIO:
                    for init_connections in INIT_CONNECTIONS_PROB:
                        for orientations_std in INIT_ORIENTATIONS_STD:
                            for emotions_mean in INIT_EMOTIONS_MEAN:
                                for emotions_std in INIT_EMOTIONS_STD:
                                    for media_conformities_mean in MEDIA_CONFORMITY_MEAN:
                                        for media_conformities_std in MEDIA_CONFORMITY_STD:
                                            config_list.append({
                                                'n_epochs': N_EPOCHS,
                                                'n_policies': n_policies,
                                                'social_sparsity': social_sparsity,
                                                'interaction_ratio': interaction_ratio,
                                                'init_connections': init_connections,
                                                'orientations_std': orientations_std,
                                                'emotions_mean': emotions_mean,
                                                'emotions_std': emotions_std,
                                                'media_conformities_mean': media_conformities_mean,
                                                'media_conformities_std': media_conformities_std,
                                                'save_to_file': save_to_file,
                                                'plot_analysis': plot_analysis
                                            })
                                            
        print(f'Starting {len(config_list)} simulations for {AGENTS_COUNT} agents and {N_EPOCHS} epochs')
        tic = time.perf_counter()
        # Create a process pool without limiting the number of processes
        process_pool = Pool()
        process_pool.map(instantiate_simulation, config_list)

        print(f'{len(config_list)} simulations for {AGENTS_COUNT} agents and {N_EPOCHS} epochs completed in {datetime.timedelta(seconds=(time.perf_counter()-tic))}')



    elif running_mode == STATISTICAL_ANALYSIS:
        config_list = []
        
        # Analysing variation in emotions
        parameter_values = np.arange(0, 1.1, 0.1)
        for emotions_mean in parameter_values:
            for _ in range(24):
                config_list.append({
                    'n_epochs': N_EPOCHS,
                    'n_policies': 7,
                    'social_sparsity': 0.6,
                    'interaction_ratio': 0.5,
                    'init_connections': 0.1,
                    'orientations_std': 0.15,
                    'emotions_mean': emotions_mean,
                    'emotions_std': 0.15,
                    'media_conformities_mean': 0,
                    'media_conformities_std': 0,
                    'save_to_file': False,
                    'plot_analysis': False
                })
                                            
        # Create a process pool without limiting the number of processes
        with Pool() as process_pool:
            model_results = process_pool.map(instantiate_simulation, config_list)
        
        stat_analysis = StatisticalAnalysis(model_results, 'emotions_mean', parameter_values)
        stat_analysis.plot_statistical_analysis()
        
        # Analysing variation in social sparsity
        parameter_values = np.arange(0, 1.1, 0.1)
        for social_sparsity in parameter_values:
            for _ in range(24):
                config_list.append({
                    'n_epochs': N_EPOCHS,
                    'n_policies': 7,
                    'social_sparsity': social_sparsity,
                    'interaction_ratio': 0.5,
                    'init_connections': 0.1,
                    'orientations_std': 0.15,
                    'emotions_mean': 0.5,
                    'emotions_std': 0.15,
                    'media_conformities_mean': 0,
                    'media_conformities_std': 0,
                    'save_to_file': False,
                    'plot_analysis': False
                })
                                            
        # Create a process pool without limiting the number of processes
        with Pool() as process_pool:
            model_results = process_pool.map(instantiate_simulation, config_list)
        
        stat_analysis = StatisticalAnalysis(model_results, 'social_sparsity', parameter_values)
        stat_analysis.plot_statistical_analysis()
        
        # Analysing variation in social sparsity
        parameter_values = np.arange(0, 0.55, 0.05)
        for media_conformities_mean in parameter_values:
            for _ in range(24):
                config_list.append({
                    'n_epochs': N_EPOCHS,
                    'n_policies': 7,
                    'social_sparsity': 0.6,
                    'interaction_ratio': 0.5,
                    'init_connections': 0.1,
                    'orientations_std': 0.15,
                    'emotions_mean': 0.5,
                    'emotions_std': 0.15,
                    'media_conformities_mean': media_conformities_mean,
                    'media_conformities_std': 0,
                    'save_to_file': False,
                    'plot_analysis': False
                })
                                            
        # Create a process pool without limiting the number of processes
        with Pool() as process_pool:
            model_results = process_pool.map(instantiate_simulation, config_list)
        
        stat_analysis = StatisticalAnalysis(model_results, 'media_conformities_mean', parameter_values)
        stat_analysis.plot_statistical_analysis()
        
        # Analysing variation in social sparsity
        parameter_values = [3, 5, 7, 10, 15, 20]
        for n_policies in parameter_values:
            for _ in range(24):
                config_list.append({
                    'n_epochs': N_EPOCHS,
                    'n_policies': n_policies,
                    'social_sparsity': 0.6,
                    'interaction_ratio': 0.5,
                    'init_connections': 0.1,
                    'orientations_std': 0.2,
                    'emotions_mean': 0.5,
                    'emotions_std': 0.15,
                    'media_conformities_mean': 0,
                    'media_conformities_std': 0,
                    'save_to_file': False,
                    'plot_analysis': False
                })
                                            
        # Create a process pool without limiting the number of processes
        with Pool() as process_pool:
            model_results = process_pool.map(instantiate_simulation, config_list)
        
        stat_analysis = StatisticalAnalysis(model_results, 'n_policies', parameter_values)
        stat_analysis.plot_statistical_analysis()
            
        
