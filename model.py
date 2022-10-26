from datetime import datetime

import numpy as np
import pickle
from sklearn.decomposition import PCA
import networkx.generators.random_graphs as random_graphs
import networkx as nx

from configuration import *
from agent import Agent



class Model:

    def __init__(self):
        # Initialise social network
        social_graph = random_graphs.erdos_renyi_graph(AGENTS_COUNT, INIT_CONNECTIONS_PROB, seed=1)
        self.adjacency_matrix = nx.to_numpy_array(social_graph)
        
        # Initialise agents
        agents_orientations = np.random.normal(INIT_ORIENTATIONS_MEAN, INIT_ORIENTATIONS_STD, AGENTS_COUNT)
        agents_orientations = np.clip(agents_orientations, ORIENTATION_MIN, ORIENTATION_MAX)
        agents_emotions = np.random.normal(INIT_ORIENTATIONS_MEAN, INIT_ORIENTATIONS_STD, AGENTS_COUNT)
        
        
        
        self.n_agents = len(agents_orientation)
        self.emotions_mean = emotions_mean
        self.emotions_std = emotions_std
        
        agents_emotion = np.random.normal(INIT_EMOTIONS_MEAN, INIT_EMOTIONS_STD, self.n_agents)
        agents_emotion = np.clip(agents_emotion, EMOTION_MIN, EMOTION_MAX)
        
        
        self.agents = [Agent(i, agents_orientation[i], agents_emotion[i], policies_orientations, self.orientations_normal) for i in range(self.n_agents)]
        
        self.noise_intensity_mean = noise_intensity_mean
        self.noise_intensity_std = noise_intensity_std
    
    
    
    
    
    def update_second_degree_adjacency(self):
        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix[i])):
                # Skip non connected agents
                if self.adjacency_matrix[i][j] == 0:
                    continue
                
                # For each connected agent j, we find all j's connection
                j_connections = np.where(self.adjacency_matrix[j] == 1)
                # Then add the second degree connection to the right column in i row
                for connection in j_connections[0]:
                    # But do not record agent's second connections to itself
                    if i != connection:
                        self.second_adjacency[i][connection] += 1
        
        
    def update_agents(self, interactions):
        """
        Performs a time step from the agents' popuylation perspective.
        n_interaction pairs will be drawn from the population, to which the first part (the active one) will update
        its opinion vector by interacting with the passive part of the pair and with ambient noise.
        After updating opinions, the agents' orientation is updated to the new opinions vector.
        
        Args:
            interactions (int): Number of agents to be updated
        """
        # We choose, from a uniform distribution, epoch_interactions pairs of agents that will interact in this epoch
        interaction_indexes = np.random.randint(0, self.n_agents, (interactions, 2))
        # We then iterate over each pair
        for indexes in interaction_indexes:
            agent = self.agents[indexes[0]]
            other = self.agents[indexes[1]]
            
            if VERBOSE:
                print("----------------------------------------")
                print(f"Interacting agent {indexes[0]} (orientation {agent.orientation:.3f}) and agent {indexes[1]} " +  
                        f"(orientation {other.orientation:.3f})\tangle:{np.degrees(agent.get_angle(other.opinions)):.1f} ")
                
            agent.interact(other.opinions)
            if VERBOSE:
                print(f"Post interaction angle: {np.degrees(agent.get_angle(other.opinions)):.1f} ")
            agent.add_noise(self.noise_opinions, self.noise_intensity)                
            agent.update_orientation(self.policies_orientations, self.orientations_normal)
            
            
    def update_noise(self):
        """
        Updates the noise opinion vector and intensity.
        """
        # First update trend opinions
        noise_orientation = np.random.uniform(ORIENTATION_MIN, ORIENTATION_MAX)
        self.noise_opinions = []
        for policy_orientation in self.policies_orientations:
            # Mean of agent's opinion on the policy should be 1 if orientation distance is 0, and -1 if 2
            orientation_distance = np.abs(noise_orientation - policy_orientation)
            opinion_mean = 0.5 * (orientation_distance - 2)**2 - 1
            # Standard deviation on agent's opinion will depend on how radical the policy orientation is
            # should be narrow for values closer to abs 1 (0.1) and wider for close to 0 (0.3)
            opinion_std = 0.1 - 0.05 * np.abs(policy_orientation) 
            # Now add an opinion
            self.noise_opinions.append(np.random.normal(opinion_mean, opinion_std))   
        # Make sure we clip to limits
        self.noise_opinions = np.clip(self.noise_opinions, OPINION_MIN, OPINION_MAX)
        # Then update intensity
        self.noise_intensity = np.random.normal(self.noise_intensity_mean, self.noise_intensity_std)
        # Make sure we clip to limits
        self.noise_intensity = np.clip(self.noise_intensity, NOISE_MIN, NOISE_MAX)
                
            
    def summon_radical(self, radicals_count, radical_side):
        """
        Summons radicals from the population by selecting the radicals_count agents that have their
        orientation value closest to radical_side. The selected agents then have their orientation set to radical_side value,
        affectiveness set to close to boundary value and opinions reinitialised based on new parameters.

        Args:
            radicals_count (int): The number of radicals to summon
            radical_side (enum): The side that the radicals should be summoned to (-1, 1)
        """
        orientations_array = np.array([self.agents[i].orientation * radical_side for i in range(self.n_agents)])
        radicals = orientations_array.argsort()[-radicals_count:][::-1]
        
        for id in radicals:
            radical = self.agents[id]
            radical.orientation = radical_side
            radical.emotion = np.random.normal(1, 0.05)
            radical.emotion = np.clip(radical.emotion, EMOTION_MIN, EMOTION_MAX)
            radical.initialise_opinions(self.policies_orientations, self.policies_orientations)
            
            
    def get_pca_snapshot(self):
        """
        Retrieves a 2-PCA analysis of the concatenation of all agents' opinion vectors

        Returns:
            [array of floats, float]: The 2-PCA components array and the explained variance ratio
        """
        opinions_list = [ self.agents[i].opinions for i in range(self.n_agents) ]
        opinion_array = np.array(opinions_list)
        pca = PCA(n_components=2, random_state=0, svd_solver="full")
        components = pca.fit_transform(opinion_array)
        return components, pca.explained_variance_ratio_

            
    def run_simulation(self, n_epochs, interaction_ratio=0.05, N_SNAPSHOTS=0, radicals=None, save_data=False):
        """
        Actually runs the whole simulation process by iterating on given epochs, updating agents and noise,
        and saving snapshots or summining radicals when required

        Args:
            n_epochs (int): number of epochs to run the simulation for
            interaction_ratio (float, optional): The ratio of the agents' population that will interact during each epoch. Defaults to 0.05.
            N_SNAPSHOTS (int, optional): Number of state snapshots to take during the simulation. Defaults to 0.
            radicals (dict, optional): Dictionary with epoch, count and side info of instructions to summon radicals. Defaults to None.
            save_data (bool, optional): Whether to save simulation data to file. Defaults to False.

        Returns:
            [dict]: The simulation info dictionary with metadata and snapshots taken
        """
        epoch_interactions = int(self.n_agents * interaction_ratio)
        snapshot_epochs = np.linspace(0, n_epochs, N_SNAPSHOTS).astype(int)
        self.snapshots = []
        print(f"Running simulation for {n_epochs} epochs with {epoch_interactions} interactions/epoch. Snapshots will be taken in {snapshot_epochs}")
        
        for epoch in range(0, n_epochs + 1):
            # Check if we need to summon radicals in the population
            if radicals and epoch in radicals["epochs"]:
                radical_index = radicals["epochs"].index(epoch)
                self.summon_radical(radicals["counts"][radical_index], radicals["sides"][radical_index])
                
            # Store snapshots for animations
            if epoch in snapshot_epochs:
                pca_components, pca_variance = self.get_pca_snapshot()

                self.snapshots.append({
                    "epoch": epoch,
                    "pca": pca_components,
                    "pca_variance": pca_variance,
                    "orientations": [self.agents[i].orientation for i in range(self.n_agents)],
                    "neg_ratio": np.sum([int(self.agents[i].orientation  < 0) for i in range(self.n_agents)]) / self.n_agents * 100.0
                })              
            
            # Step noise and agents
            self.update_noise()
            self.update_agents(epoch_interactions)
        
            # Print status on each 10%
            if epoch % (n_epochs / 10) == 0:
                print(f"{int(epoch/n_epochs * 100)}%")
            
        # Build the simulation info
        simulation_info = {
            "n_agents": self.n_agents,
            "n_policies": self.n_policies,
            "n_epochs": n_epochs,
            "policies_neg": self.policies_neg,
            "policies_mean": self.policies_mean,
            "policies_std": self.policies_std,
            "interaction_ratio": interaction_ratio,
            "emotions_mean": self.emotions_mean,
            "emotions_std": self.emotions_std,
            "noise_intensity_mean": self.noise_intensity_mean,
            "noise_intensity_std": self.noise_intensity_std,
            "snapshots": self.snapshots
        }
        # Save snaphsots to file
        if save_data:
            timestamp = datetime.now().strftime("%m-%d-%H.%M")
            filename = DATA_DIR + timestamp + f"-run-({self.n_agents}|{n_epochs}|{self.n_policies}|{self.policies_neg}|{self.policies_mean}|{self.policies_std}|{interaction_ratio}|{N_SNAPSHOTS}|{self.emotions_mean:.2f}|{self.emotions_std:.2f}|{self.noise_intensity_mean:.2f}|{self.noise_intensity_std:.2f}).dat"
            print(f"Saving snapshots data to: {filename}")
            pickle.dump(simulation_info, open(filename, "wb"))
            
        return simulation_info