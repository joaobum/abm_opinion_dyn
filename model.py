import numpy as np
from sklearn.decomposition import PCA
import networkx.generators.random_graphs as random_graphs
import networkx as nx

from configuration import *
from agent import Agent

class Model:

    def __init__(self,
                 n_policies,
                 social_sparsity,
                 interaction_ratio,
                 init_connections,
                 orientations_std,
                 emotions_mean,
                 emotions_std,
                 media_conformities_mean,
                 media_conformities_std):

        # Initialise social network
        self.social_sparsity = social_sparsity
        self.interaction_ratio = interaction_ratio
        self.init_connections_prob = init_connections
        self.social_graph = random_graphs.erdos_renyi_graph(
            AGENTS_COUNT, init_connections, seed=SEED)
        self.adjacency_matrix = nx.to_numpy_array(self.social_graph)

        # Initialise agents
        # Draw orientation, emotion and media conformity values from
        # normal distribution for all agents, then make sure to clip them to
        # boundaries
        self.n_policies = n_policies
        agents_orientations = np.random.normal(
            INIT_ORIENTATIONS_MEAN, orientations_std, AGENTS_COUNT)
        agents_orientations = np.clip(
            agents_orientations, ORIENTATION_MIN, ORIENTATION_MAX)
        # We store the values for emotions and conformities
        self.agents_emotions = np.random.normal(
            emotions_mean, emotions_std, AGENTS_COUNT)
        self.agents_emotions = np.clip(
            self.agents_emotions, EMOTION_MIN, EMOTION_MAX)
        self.media_conformities = np.random.normal(
            media_conformities_mean, media_conformities_std, AGENTS_COUNT)
        self.media_conformities = np.clip(
            self.media_conformities, MEDIA_CONFORMITY_MIN, MEDIA_CONFORMITY_MAX)

        # Populate the agents array. The agents initialisation draws values
        # for their opinion array based on initial orientation and emotion.
        self.agents = [
            Agent(i, self.n_policies, agents_orientations[i], self.agents_emotions[i],
                  self.adjacency_matrix[i], self.media_conformities[i])
            for i in range(AGENTS_COUNT)
        ]

        # Initialise dynamic aggregation arrays
        self.group_attractions = np.zeros((AGENTS_COUNT, AGENTS_COUNT))
        self.media_attractions = np.zeros((AGENTS_COUNT, MEDIA_OUTLETS_COUNT))
        self.refresh_group_opinions()
        self.refresh_media_opinions()
        # Update social attraction for all agents
        self.update_agents_attractions(range(AGENTS_COUNT))
        # Update media attraction for all agents
        self.update_media_attractions(range(AGENTS_COUNT))

        # Set model data collection dictionary
        self.data = {
            'n_policies': n_policies,
            'social_sparsity': social_sparsity,
            'interaction_ratio': interaction_ratio,
            'init_connections': init_connections,
            'orientations_std': orientations_std,
            'emotions_mean': emotions_mean,
            'emotions_std': emotions_std,
            'media_conformities_mean': media_conformities_mean,
            'media_conformities_std': media_conformities_std,
            'agents_emotions': self.agents_emotions,
            'candidates_opinions': CANDIDATES_OPINIONS_MEAN,
            'seed': SEED,
            'connections_balance': 0,
            'snapshots': []
        }
        self.connections_balance = 0

    def step(self):
        # We first select a random sample of the agent's pool without replacement
        active_agents = np.random.choice(AGENTS_COUNT, int(
            AGENTS_COUNT * self.interaction_ratio), False)
        #######################################################################
        #   1.  Trust matrices update
        #######################################################################
        #tic = time.perf_counter()
        # Based on the media and group attractions, the trust is updated
        for i in active_agents:
            self.agents[i].update_agents_trust(self.group_attractions)

        # And local arrays refreshed
        self.refresh_group_trust()
        self.refresh_media_trust()

        # print(f'1  ->  {time.perf_counter() - tic}')

        #######################################################################
        #   2.  Opinion interaction
        #######################################################################
        #tic = time.perf_counter()
        # We got all we need to calculate opinion influences
        # Get the total group and total media influence based on trust in each element
        group_influence = np.matmul(self.group_trust, self.group_opinions)
        media_influence = np.matmul(self.media_trust, self.media_opinions)

        # Then calculate total external influence, weighted by media conformity
        external_influence = (group_influence.T * (1 - self.media_conformities)).T \
            + (media_influence.T * self.media_conformities).T

        # Get noise from communications innacuracies
        comms_noise = np.random.normal(
            NOISE_MEAN, NOISE_STD, (AGENTS_COUNT, self.n_policies))
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
        for i in active_agents:
            self.agents[i].interact(external_influence[i])
            self.agents[i].add_noise(noise_influence[i])
            # Strength is normalised by the max norm in the space
            self.agents[i].opinion_strength = np.linalg.norm(
                self.agents[i].opinions) / np.sqrt(self.n_policies)

        # Refresh local opinion arrays
        self.refresh_group_opinions()
        self.refresh_media_opinions()
        # Update social attraction for active agents
        self.update_agents_attractions(active_agents)
        # Update media attraction for active agents
        self.update_media_attractions(active_agents)
        
        # print(f'2  ->  {time.perf_counter() - tic}')

        #######################################################################
        #   3.  Social network update
        #######################################################################
        #tic = time.perf_counter()
        for i in active_agents:
            for j in range(AGENTS_COUNT):
                # Attraction of i towards j
                attraction_i = self.group_attractions[i][j]
                # Attraction of j towards i
                attraction_j = self.group_attractions[j][i]

                # Attractions are modulated by emotional affectiveness, so they may not be symmetric
                # We perform a destroy or create action only if both attractions satisfy conditions
                # In case neither does or they disagree, the relationship stays as is
                if self.agents[i].adjacency[j] == 0:
                    if attraction_i > self.social_sparsity and attraction_j > self.social_sparsity:
                        self.agents[i].adjacency[j] = 1
                        self.agents[j].adjacency[i] = 1
                        self.data['connections_balance'] += 1

                # If both attractions are lower than sparsity, destroy connection
                if self.agents[i].adjacency[j] == 1:
                    if attraction_i < self.social_sparsity and attraction_j < self.social_sparsity:
                        self.agents[i].adjacency[j] = 0
                        self.agents[j].adjacency[i] = 0
                        self.data['connections_balance'] -= 1
                    
        # Update the adjacency matrix
        self.refresh_social_graph()
        # print(f'3  ->  {time.perf_counter() - tic}')

    def refresh_group_opinions(self):
        self.group_opinions = np.array(
            [agent.opinions for agent in self.agents]
        )
        self.group_opinion_strengths = np.array(
            [agent.opinion_strength for agent in self.agents]
        )

        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tGroup opinions refreshed:')
            print_array(self.group_opinions)
            print_array(self.group_opinion_strengths)


    def update_agents_attractions(self, active_agents):
        # Iterate over all active agents
        for active_agent in active_agents:
            for reference_agent in range(AGENTS_COUNT):
                # Ignore the self-referencing diagonal
                if active_agent == reference_agent:
                    continue
                # And calculate the attraction of active_agent towards reference_agent
                self.group_attractions[active_agent][reference_agent] = \
                    self.agents[active_agent].get_social_attraction(
                        self.group_opinions[reference_agent],
                        self.social_graph.degree[active_agent] / AGENTS_COUNT,
                        self.social_graph.degree[reference_agent] / AGENTS_COUNT
                )

    def refresh_group_trust(self):
        self.group_trust = np.array(
            [agent.agents_trust for agent in self.agents]
        )
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tGroup trust refreshed:')
            print_array(self.group_trust)

    def refresh_media_opinions(self):
        # Each media outlet will draw an opinion value for each policy
        # from a normal distribution, with the mean representing the outlet's
        # main orientation, and std is the flexibility
        self.media_opinions = np.array(
            [
                np.random.normal(
                    MEDIA_OUTLETS_MEANS[i],
                    MEDIA_OUTLETS_STD[i],
                    self.n_policies
                ) for i in range(MEDIA_OUTLETS_COUNT)
            ]
        )
        self.media_opinions = np.clip(
            self.media_opinions, OPINION_MIN, OPINION_MAX)
        # Strengths are the norms normalised by max norm
        self.media_opinion_strengths = np.linalg.norm(
            self.media_opinions, axis=1) / np.sqrt(self.n_policies)

        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tMedia opinions refreshed:')
            print_array(self.media_opinions)

    def update_media_attractions(self, active_agents):
        # Iterate over all active agents
        for agent in active_agents:
            for media in range(MEDIA_OUTLETS_COUNT):
                # And calculate the attraction of agent towards media
                self.media_attractions[agent][media] = \
                    self.agents[agent].get_social_attraction(self.media_opinions[media])      

    def refresh_media_trust(self):
        # Media trust is calculated by row normalising the attraction values
        self.media_trust = self.media_attractions / np.sum(self.media_attractions, axis=1)[:,None]
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tMedia trust refreshed:')
            print_array(self.media_trust)

    def refresh_social_graph(self):
        self.adjacency_matrix = np.array(
            [agent.adjacency for agent in self.agents]
        )
        self.social_graph = nx.from_numpy_matrix(self.adjacency_matrix)
        
        if VERBOSITY & V_MODEL:
            print('****************************************')
            print('*\tAdjacency matrix refreshed:')
            print_array(self.adjacency_matrix)

    def get_election_poll(self):
        # The poll is based on drawing candidate's opinions for each poll
        candidates_opinions = np.array(
            [
                np.random.normal(
                    CANDIDATES_OPINIONS_MEAN[i],
                    CANDIDATES_OPINIONS_STD[i],
                    self.n_policies
                ) for i in range(CANDIDATES_COUNT)
            ]
        )
        candidates_opinions = np.clip(
            candidates_opinions, OPINION_MIN, OPINION_MAX)
        
        candidate_attractions = np.zeros((AGENTS_COUNT, CANDIDATES_COUNT))
        # All agents have a voice
        for agent in range(AGENTS_COUNT):
            for candidate in range(CANDIDATES_COUNT):
                # And calculate the attraction of agent towards each candidate
                candidate_attractions[agent][candidate] = \
                    self.agents[agent].get_social_attraction(candidates_opinions[candidate])  

        # # Poll is based on agents choosing the candidate they're most
        # # attracted to
        # agents_votes = list(np.argmax(candidate_attractions, axis=1))
        # vote_count = [agents_votes.count(i) for i in range(CANDIDATES_COUNT)]
        # poll = vote_count / np.sum(vote_count)
        
        # The poll is the normalised mean of attractions
        mean_attractions = np.mean(candidate_attractions, axis=0)
        poll = mean_attractions / np.sum(mean_attractions)

        return poll

    def run(self, n_epochs: int):
        snapshot_epochs = np.linspace(0, n_epochs, N_SNAPSHOTS).astype(int)
        print('\n********************************************************************************\n')
        print(f'Starting model run for {n_epochs} epochs.\n')

        # Start epoch from 0 so we save the initial state before stepping
        for epoch in range(0, n_epochs + 1):
            # Print status on each 10%
            if epoch % (n_epochs / 10) == 0:
                print(f'Run progress:\t{int(epoch/n_epochs * 100)}%')

            # Store snapshots for plotting
            if epoch in snapshot_epochs:
                poll = self.get_election_poll()
                self.data['snapshots'].append({
                    'epoch': epoch,
                    'group_opinions': self.group_opinions.copy(),
                    'group_attractions': self.group_attractions.copy(),
                    'media_opinions': self.media_opinions.copy(),
                    'adjacency': self.adjacency_matrix.copy(),
                    'poll': poll
                })

            # Step the model
            self.step()
