import math

import numpy as np

from configuration import *


def normpdf(x, sd):
    return math.exp(-((x/sd)**2)/2) / (sd * math.sqrt(2*math.pi))


class Agent:
    def __init__(self, id: int, n_policies: int, orientation: float, emotion: float, adjacency: np.array, media_conformity: float) -> None:
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
            orientation, ((1 - emotion)*(1 - (orientation + 1) / 2)), n_policies)
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)
        # Opinion strength is normalised by max norm in the space
        self.opinion_strength =  np.linalg.norm(
            self.opinions) / np.sqrt(len(self.opinions))

        # Initialise zero arrays that will be updated on each iteration
        self.agents_trust = np.zeros(AGENTS_COUNT)
        self.media_trust = np.zeros(MEDIA_OUTLETS_COUNT)
        

    def update_agents_trust(self, group_attractions: np.ndarray):
        # Clean the current array
        self.agents_trust = np.zeros(AGENTS_COUNT)
        # Get the indexes of relevant opinions from the adjacency matrix
        connections_indexes = self.adjacency.nonzero()[0]
        
        # If we don't have any connections, keep the zeros
        if len(connections_indexes) > 0:
            # Iterate in all connections storing the normal function for the dissimilarity
            for i in connections_indexes:
                self.agents_trust[i] = group_attractions[self.id][i]

            # And finally normalise the trust
            self.agents_trust = self.agents_trust / np.sum(self.agents_trust)


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
    
    def get_social_attraction(self, ref_opinions, self_degree=1, ref_degree=1):
        # The gravitational constant is defined by
        g = (1 - self.emotion) * (1 - min(0.95, self.opinion_strength))
        # Directional distance is simply the similarity inverted in the range [0, 1]
        directional_distance = 1 - self.get_directional_similarity(ref_opinions) 
        
        attraction_mass = (1 + self_degree) / 2
        ref_attraction_mass = (1 + ref_degree) / 2
        
        # Attraction is based on a gravitation law. With the directional distance
        # reverse squared and the masses are the opinion centralisty attractiveness
        # The gravitanional constant is modulated by the emotion affectiveness
        attraction = math.exp(-(directional_distance ** 2) / (g * attraction_mass * ref_attraction_mass))
        
        return attraction

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
            reference_strength = np.linalg.norm(ref_opinions) / np.sqrt(len(ref_opinions))
            own_strength = np.linalg.norm(self.opinions) / np.sqrt(len(ref_opinions))
            relative_strength = np.clip(reference_strength / own_strength, 0.5, 2)
            dissonance_factor = np.sin(2 * angle) / 2
            rotation_angle = angle * dissonance_factor * relative_strength
            # And do the job
            if VERBOSITY & V_AGENT:
                print(
                    f'Interacting: angle = {np.degrees(angle):.2f}\tdissonance_factor = {dissonance_factor:.3f}\trelative_strength = {relative_strength:.3f}\trotation_angle = {np.degrees(rotation_angle):.2f}')
            self.rotate_opinions(ref_opinions, rotation_angle)

    def add_noise(self, noise):
        # Just add the noise, it was already attenuated by emotions in the model scope
        self.opinions += noise
        # Make sure we're still in range
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)

    
