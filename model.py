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
        social_graph = random_graphs.erdos_renyi_graph(
            AGENTS_COUNT, INIT_CONNECTIONS_PROB, seed=1)
        self.adjacency_matrix = nx.to_numpy_array(social_graph)
        
        # Initialise agents
        # Draw orientation, emotion and media conformity values from
        # normal distribution for all agents, then make sure to clip them to
        # boundaries
        agents_orientations = np.random.normal(
            INIT_ORIENTATIONS_MEAN, INIT_ORIENTATIONS_STD, AGENTS_COUNT)
        agents_orientations = np.clip(
            agents_orientations, ORIENTATION_MIN, ORIENTATION_MAX)
        # We store the values for emotions and conformities
        self.agents_emotions = np.random.normal(
            INIT_EMOTIONS_MEAN, INIT_EMOTIONS_STD, AGENTS_COUNT)
        self.agents_emotions = np.clip(self.agents_emotions, EMOTION_MIN, EMOTION_MAX)
        self.media_conformities = np.random.normal(
            MEDIA_CONFORMITY_MEAN, MEDIA_CONFORMITY_STD, AGENTS_COUNT)
        self.media_conformities = np.clip(
            self.media_conformities, MEDIA_CONFORMITY_MIN, MEDIA_CONFORMITY_MAX)
        
        # Populate the agents array. The agents initilisation draws values
        # for their opinion array based on initial orientation and emotion.
        self.agents = [
            Agent(i, agents_orientations[i], self.agents_emotions[i],
                  self.adjacency_matrix[i], self.media_conformities[i])
            for i in range(AGENTS_COUNT)
        ]
        
        # Refreshing the second degree adjacency also updates agents
        self.update_second_degree_adjacency()

        # Initialise dynamic aggregation arrays
        # We first refresh opinions local arrays
        self.refresh_group_opinions()
        self.refresh_media_opinions()
        # Based on the media and group opinions, the trust is updated
        for agent in self.agents:
            agent.update_agents_trust(self.group_opinions)
            agent.update_media_trust(self.media_opinions)
        # And local arrays refreshed
        self.refresh_group_trust()
        self.refresh_media_trust()

        self.epoch = 0

    def step(self):
        #######################################################################
        #   1.  Opinion interaction
        #######################################################################
        # We got all we need to calculate opinion influences
        # Get the total group and total media influence based on trust in each element
        group_influence = np.matmul(self.group_trust, self.group_opinions)
        media_influence = np.matmul(self.media_trust, self.media_opinions)

        # Then calculate total external influence, weighted by media conformity
        external_influence = (group_influence.T * (1 - self.media_conformities)).T \
            + (media_influence.T * self.media_conformities).T

        # Get noise from communications innacuracies
        comms_noise = np.random.normal(
            NOISE_MEAN, NOISE_STD, (AGENTS_COUNT, POLICIES_COUNT))
        noise_influence = (comms_noise.T * (1 - self.agents_emotions)).T

        # Print state if required
        if VERBOSITY & V_MODEL:
            print('\n* Group influence:')
            print_array(group_influence)
            print('\n* Media influence:')
            print_array(media_influence)
            print('\n* Media conformities:')
            print_array(self.media_conformities)
            print('\n* Group influence updated:')
            print_array((group_influence.T * (1 - self.media_conformities)).T)
            print('\n* Media influence updated:')
            print_array((media_influence.T * self.media_conformities).T)
            print('\n* External influence:')
            print_array(external_influence)
            print('\n* Noise influence:')
            print_array(noise_influence)

        # Now update each agent's opinion by making it interact with the calculated
        # external influence. The opinions will also suffer some noise that represents
        # communication innacuracies.
        for i in range(AGENTS_COUNT):
            self.agents[i].interact(external_influence[i])
            self.agents[i].add_noise(noise_influence[i])
            
        # Refresh local opinion arrays
        self.refresh_group_opinions()
        self.refresh_media_opinions()

        #######################################################################
        #   2.  Social network update
        #######################################################################
        # Update the adjacency matrices
        self.refresh_adjacency_matrix()
        self.update_second_degree_adjacency()
        
        # Based on similarities with first and second degree connections, the
        # agents social network gets updated
        for agent in self.agents:
            agent.update_agents_similarities(self.group_opinions)
            create_connections, end_connections = agent.update_social_network()
            # End the connections symmetrically 
            for connection in end_connections:
                add_destruction()
                self.agents[agent.id].adjacency[connection] = 0
                self.agents[connection].adjacency[agent.id] = 0
            # Create new connections symmetrically 
            for connection in create_connections:
                add_creation()
                self.agents[agent.id].adjacency[connection] = 1
                self.agents[connection].adjacency[agent.id] = 1
        
        #######################################################################
        #   3.  Trust matrices update
        #######################################################################
        # Based on the media and group opinions, the trust is updated
        for agent in self.agents:
            agent.update_agents_trust(self.group_opinions)
            agent.update_media_trust(self.media_opinions)
            
        # And local arrays refreshed
        self.refresh_group_trust()
        self.refresh_media_trust()


    def refresh_group_opinions(self):
        self.group_opinions = np.array(
            [agent.opinions for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tGroup opinions refreshed:')
            print_array(self.group_opinions)

    def refresh_group_trust(self):
        self.group_trust = np.array(
            [agent.agents_trust for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tGroup trust refreshed:')
            print_array(self.group_trust)

    def refresh_media_opinions(self):
        self.media_opinions = np.array(
            [
                np.random.normal(
                    MEDIA_OUTLETS_MEANS[i],
                    MEDIA_OUTLETS_STD[i],
                    POLICIES_COUNT
                ) for i in range(MEDIA_OUTLETS_COUNT)
            ]
        )
        self.media_opinions = np.clip(
            self.media_opinions, OPINION_MIN, OPINION_MAX)

        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tMedia opinions refreshed:')
            print_array(self.media_opinions)

    def refresh_media_trust(self):
        self.media_trust = np.array(
            [agent.media_trust for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tMedia trust refreshed:')
            print_array(self.media_trust)
    
    def refresh_adjacency_matrix(self):
        self.adjacency_matrix = np.array(
            [agent.adjacency for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tAdjacency matrix refreshed:')
            print_array(self.adjacency_matrix)

    def update_second_degree_adjacency(self):
        # Reset current state
        self.second_adjacency = np.zeros((AGENTS_COUNT, AGENTS_COUNT))
        # Iterate all rows of the adjacency matrix
        for i in range(AGENTS_COUNT):
            # Run though all columns for possible connected agents
            for j in range(AGENTS_COUNT):
                # Skip non connected agents
                if self.adjacency_matrix[i][j] == 0:
                    continue
                # For each connected agent j, we find all j's connection
                j_connections = np.where(self.adjacency_matrix[j] == 1)
                # Then add the second degree connection to the right column in i row
                for connection in j_connections[0]:
                    # But do not record agent's second connections to itself,
                    # or record second connections of who's directly connected
                    if i != connection and self.adjacency_matrix[i][connection] == 0:
                        self.second_adjacency[i][connection] += 1
            # Then update at agent level
            self.agents[i].second_adjacency = self.second_adjacency[i]

    def get_election_poll(self):
        # Initialise a zero array
        poll = np.zeros(CANDIDATES_COUNT)
        # Sum the normalised intentions of all agents
        for agent in self.agents:
            poll += agent.get_vote_intention()
        # Then normalise the sum
        poll = poll / np.sum(poll)
        return poll

    def get_pca_snapshot(self):
        opinions_list = [self.agents[i].opinions for i in range(self.n_agents)]
        opinion_array = np.array(opinions_list)
        pca = PCA(n_components=2, random_state=0, svd_solver='full')
        components = pca.fit_transform(opinion_array)
        return components, pca.explained_variance_ratio_

    def run(self, n_epochs: int):
        snapshot_epochs = np.linspace(0, n_epochs, N_SNAPSHOTS).astype(int)
        data_snapshots = []
        print('\n********************************************************************************\n')
        print(f'Starting model run for {n_epochs} epochs.\n')

        # Start epoch from 0 so we save the initial state before stepping
        for epoch in range(0, n_epochs + 1):
            # Print status on each 10%
            if epoch % (n_epochs / 10) == 0:
                print(f'Run progress:\t{int(epoch/n_epochs * 100)}%')

            # Store snapshots for plotting
            if epoch in snapshot_epochs:
                # pca_components, pca_variance = self.get_pca_snapshot()
                poll = self.get_election_poll()
                data_snapshots.append({
                    'epoch': epoch,
                    'group_opinions': self.group_opinions,
                    'media_opinions': self.media_opinions,
                    'adjacency': self.adjacency_matrix,
                    'poll': poll
                })

            # Step the model
            self.step()

        # Build the simulation info
        simulation_info = {
            'snapshots': data_snapshots
        }

        return simulation_info
