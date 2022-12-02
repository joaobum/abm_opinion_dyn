import numpy as np

from configuration import *


def normpdf(x, sd):
    return math.exp(-((x/sd)**2)/2) / (sd * math.sqrt(2*math.pi))


class Agent:
    def __init__(self, id: int, orientation: float, emotion: float, adjacency: np.array, media_conformity: float) -> None:
        if VERBOSITY & V_AGENT:
            print(
                f'Initialising agent {id}:\torientation = {orientation:.3f}\temotion = {emotion:.3f}')
        # Store arguments
        self.id = id
        self.emotion = emotion
        self.adjacency = adjacency
        self.media_conformity = media_conformity

        # Initialise opinions based on initial orientation and emotional affectiveness
        # Agents that are more emotional about their opinions will have a narrower distribution
        # around their orientation. Orientation gets normalised from [-1, 1] to [0, 1]
        self.opinions = np.random.normal(
            orientation, ((1 - emotion)*(1 - (orientation + 1) / 2)), POLICIES_COUNT)
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)

        # Initialise zero arrays that will be updated on each iteration
        self.agents_similarities = np.zeros(AGENTS_COUNT)
        self.agents_trust = np.zeros(AGENTS_COUNT)
        self.media_trust = np.zeros(MEDIA_OUTLETS_COUNT)
        self.media_similarities = np.zeros(AGENTS_COUNT)

    def print_state(self):
        print('\n****************************************')
        print(f'\nAgent {self.id}\n state:')
        print(f'******************\nEmotion = {self.emotion}')
        print(
            f'******************\nMedia conformity = {self.media_conformity}')
        print('******************\nAdjacency')
        print_array(self.adjacency)
        print('******************\nGroup Dissimilarities')
        print_array(1 - self.agents_similarities)
        print('******************\nGroup trust')
        print_array(self.agents_trust)
        print('******************\nMedia Dissimilarities')
        print_array(1 - self.media_similarities)
        print('******************\nMedia Trust')
        print_array(self.media_trust)
        print('\n****************************************')
        

    def update_agents_trust(self, group_similarities: np.ndarray):
        # Clean the current array
        self.agents_trust = np.zeros(AGENTS_COUNT)
        # Get the indexes of relevant opinions from the adjacency matrix
        connections_indexes = self.adjacency.nonzero()[0]
        # If we don't have any connections, keep the zeros
        if len(connections_indexes) > 0:
            # Variance of the trust distribution is based on emotion
            # Agents with high emotional affectiveness on their opinon will tend to trust
            # mostly agents that think similarly.
            trust_variance = 1.05 - self.emotion
            # Iterate in all connections storing the normal function for the dissimilarity
            for i in connections_indexes:
                # Dissimilarity = 1 - similarity
                self.agents_trust[i] = normpdf(
                    1 - group_similarities[self.id][i], trust_variance)

            # And finally normalise the trust
            self.agents_trust = self.agents_trust / np.sum(self.agents_trust)

    def update_media_similarities(self, media_opinions: np.ndarray):
        # Clean the current array
        self.media_similarities = np.zeros(MEDIA_OUTLETS_COUNT)
        # Iterate in all connections, getting their directional similarity
        for i in range(MEDIA_OUTLETS_COUNT):
            # Dissimilarity = 1 - similarity
            self.media_similarities[i] = self.get_directional_similarity(
                media_opinions[i])

    def update_media_trust(self, media_opinions: np.ndarray):
        # Clean the current array
        self.media_trust = np.zeros(MEDIA_OUTLETS_COUNT)
        # Variance of the trust distribution is based on emotion like for agents
        trust_variance = 1.05 - self.emotion
        # Update similarity array
        self.update_media_similarities(media_opinions)
        # Iterate in all outlets storing the normal function for the dissimilarity
        for i in range(MEDIA_OUTLETS_COUNT):
            # Dissimilarity = 1 - similarity
            self.media_trust[i] = normpdf(
                1 - self.media_similarities[i], trust_variance)

        # And finally normalise the trust
        self.media_trust = self.media_trust / np.sum(self.media_trust)

    def update_social_network(self, group_similarities: np.ndarray, group_opinion_strengths: np.ndarray):
        # Initialise an array of potential connection breaks
        end_connections = []
        # Get first connections
        connections_indexes = self.adjacency.nonzero()[0]
        # Now check if any of the existing connections need undoing
        for i in connections_indexes:
            # The probability of endind an existing relationship is influenced
            # by the dissimilarity and emotional affectiveness
            end_prob = (1 - group_similarities[self.id][i]) * self.emotion * group_opinion_strengths[self.id] * group_opinion_strengths[i]
            add_end_prob(end_prob)
            # Then we draw from that probability, and in case true we end
            # the adjacency and store the value to return so we end the adjacency
            # on the other agent as well
            if np.random.choice([True, False], p=[end_prob, 1-end_prob]):
                end_connections.append(i)
        # Analogous behaviour for new connections
        # Initialise an array of potential new connections
        create_connections = []

        # Grab random from population, excluding ourselves and whos already connected
        eval_indexes = [id for id in range(AGENTS_COUNT)
                if id not in np.append(connections_indexes, self.id)]
        if len(eval_indexes):
            create_evals = np.random.choice(
                eval_indexes,
                int(10)
            )

            # Now check if any of the existing connections need undoing
            for i in create_evals:
                # The probability of creating a new connection is proportional
                # to similarity and the ratio of possible second degree connections
                # (-2 discounts the direct connection and the agent itself)
                create_prob = group_similarities[self.id][i] * self.emotion * group_opinion_strengths[self.id] * group_opinion_strengths[i]
                add_create_prob(create_prob)
                # Then we draw from that probability, and in case true we end
                # the adjacency and store the value to return so we end the adjacency
                # on the other agent as well
                if np.random.choice([True, False], p=[create_prob, 1-create_prob]):
                    create_connections.append(i)

        # Now return the arrays so we can end/create the connection on the other involved agent
        return create_connections, end_connections

    def get_angle(self, ref_opinions):
        '''
        Retrieves the angle between the agent's opinion array and the one in the other_opinons parameter.
        Angle calculated by the cosine of the dot product.

        Args:
            ref_opinions (array of floats): The reference array of opinions for the agent's angle to be calculated on

        Returns:
            float: The angle in radians between the agent's opinion and the ref_opinions parameter
        '''
        cos = np.dot(self.opinions, ref_opinions) / \
            (np.linalg.norm(self.opinions) * np.linalg.norm(ref_opinions))
        cos = np.clip(cos, -1, 1)
        return np.arccos(cos)

    def get_directional_similarity(self, ref_opinions):
        return 1 - (self.get_angle(ref_opinions) / np.pi)

    def rotate_opinions(self, ref_opinions, angle):
        '''
        Rotates the agent's opinion angle towards the other_opinons parameter by the angle parameter amount.

        Args:
            ref_opinions (array of floats): The reference array of opinions for the agent to rotate towards
            angle (float): The rotation angle in radians
        '''
        if np.abs(angle) < 1e-5:
            if VERBOSITY & V_AGENT:
                print(f'Rotation angle is too small ({angle}), ignoring...')
            return

        # Gram-Schmidt orthogonalization
        versor_self = self.opinions / np.linalg.norm(self.opinions)
        axis = ref_opinions - np.dot(versor_self, ref_opinions) * versor_self
        versor_axis = axis / np.linalg.norm(axis)

        # Perform rotation
        I = np.identity(len(self.opinions))
        R = I + (np.outer(versor_axis, versor_self) - np.outer(versor_self, versor_axis)) * np.sin(angle) + \
            (np.outer(versor_self, versor_self) +
             np.outer(versor_axis, versor_axis)) * (np.cos(angle)-1)
        self.opinions = np.matmul(R, self.opinions)

        # Make sure we're still in range
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)

    def interact(self, ref_opinions):
        '''
        Performs the agent's interaction with the other_opinons parameter.
        The rotation angle is calculaated by attenuating the totaal angle between the agents by
        a dissonance factor ang the agent's emotional attachment.
        A rotation is there performed by the amount of the calculated rotation_angle towards ref_opinions.

        Args:
            ref_opinions (array of floats): The array of opinions for the agent to interact with.
        '''
        # Check if there's any influence
        if np.any(ref_opinions):
            # Get the relative angle
            angle = self.get_angle(ref_opinions)
            # The relative strength of opinions dictate the bias of opinion rotation
            reference_strength = np.linalg.norm(ref_opinions) / np.sqrt(POLICIES_COUNT)
            own_strength = np.linalg.norm(self.opinions) / np.sqrt(POLICIES_COUNT)
            relative_strength = np.clip(reference_strength / own_strength, 0.5, 2)
            dissonance_factor = np.sin(2 * angle) / 2
            rotation_angle = angle * dissonance_factor * relative_strength * (1 - self.emotion)
            # And do the job
            if VERBOSITY & V_AGENT:
                print(
                    f'Interacting: angle = {np.degrees(angle):.2f}\tdissonance_factor = {dissonance_factor:.3f}\temotion = {self.emotion:.3f}\trotation_angle = {np.degrees(rotation_angle):.2f}')
            self.rotate_opinions(ref_opinions, rotation_angle)

    def add_noise(self, noise):
        # Just add the noise, it was already attenuated by emotions in the model scope
        self.opinions += noise
        # Make sure we're still in range
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)

    def get_vote_intention(self):
        # Get the similarity to each candidate's position
        candidates_similarities = []
        for i in range(CANDIDATES_COUNT):
            # The randomness is due to each agent perceiving the candidate's opinion with
            # some variation.
            perceived_position = np.random.normal(
                CANDIDATES_OPINIONS_MEAN[i],
                CANDIDATES_OPINIONS_STD[i],
                POLICIES_COUNT
            )
            candidates_similarities.append(
                self.get_directional_similarity(perceived_position))
        # The vote probabilities is the opitinons similarities normalised
        vote_intention = candidates_similarities / \
            np.sum(candidates_similarities)
        return vote_intention
