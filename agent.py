###############################################################################
#   University of Sussex - Department of Informatics
#   MSc in Artificial Intelligence and Adaptive Systems
#
#   Project title: A co-evolution model for opinions in a social network
#   Candidate number: 229143
#
###############################################################################

# Standard libraries
import math
# External packages
import numpy as np
# Internal modules
from configuration import *


class Agent:
    """
    Definition of the Agent class.
    Each model is composed by an array of AGENTS_COUNT agents. This class 
    Initialises the agent's defining properties and provide methods for opinion
    interaction and social attraction and trust calculation.
    """

    def __init__(
            self,
            id: int,
            n_policies: int,
            orientation: float,
            emotion: float,
            media_conformity: float,
            adjacency: np.array) -> None:
        """
        Initialises an instance of the Agent class.
        Stores the values for emotions and media conformity and initialises the 
        agent's opinion array based on orientation and emotion.
        
        Arguments:
            id {int} -- The agent’s identifier (index in the Model class array).
            n_policies {int} -- Number of policies in which the agent should have an opinion.
            (i.e., dimensionality of the opinion-space)
            orientation {float} -- Value used as a mean opinion value and modulator 
            for standard deviation in opinions initialisation.
            emotion {float} -- The agent's emotional bias.
            media_conformity {float} -- The ratio by which the agent's beliefs 
            are influenced by the media.
            adjacency {np.array} -- The row of the adjacency matrix representing 
            the agent's connections
        """
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
            orientation, ((1 - emotion)*(1 - (abs(orientation) + 1) / 2)), n_policies)
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)
        # Opinion strength is normalised by max norm in the space
        self.opinion_strength = np.linalg.norm(
            self.opinions) / np.sqrt(len(self.opinions))

        # Initialise zero arrays that will be updated on each iteration
        self.agents_trust = np.zeros(AGENTS_COUNT)
        self.media_trust = np.zeros(MEDIA_OUTLETS_COUNT)

    def update_agents_trust(self, group_attractions: np.ndarray) -> None:
        """
        Updates the agent's trust array in other agents.
        The trust value is given by the normalisation of social attraction
        values for all connections.

        Arguments:
            group_attractions {np.ndarray} -- The group's social attraction
            matrix to be used in trust normalisation.
        """
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

    def get_angle(self, ref_opinions: np.ndarray) -> float:
        """
        Calculated via dot product rule the relative angle between the agent's
        opinions and the ref_opinions vectors.

        Arguments:
            ref_opinions {np.ndarray} -- The reference array of opinions for 
            calculating the relative angle

        Returns:
            float -- The angle in radians between the agent's opinion and the ref_opinions parameter
        """
        cos = np.dot(self.opinions, ref_opinions) / \
            (np.linalg.norm(self.opinions) * np.linalg.norm(ref_opinions))
        cos = np.clip(cos, -1, 1)
        return np.arccos(cos)

    def get_social_attraction(self, 
                              ref_opinions: np.ndarray, 
                              self_degree: int = 1, 
                              ref_degree: int = 1
                              ) -> float:
        """
        Calculates the gravity-based social attraction between the agent and the
        opinion vector ref_opinions. The relative connection degrees can also be 
        used in case of social attraction.

        Arguments:
            ref_opinions {np.ndarray} -- Vector of reference opinions for the 
            attraction to be calculated against.

        Keyword Arguments:
            self_degree {int} -- Relative connections degree for the agent 
            in case of social attraction (default: {1})
            ref_degree {int} -- Relative connections degree for the reference 
            in case of socia attraction (default: {1})

        Returns:
            float -- _description_
        """
        # The gravitational constant is defined by
        g = (1 - self.emotion) * (1 - min(0.95, self.opinion_strength))
        # Directional distance is simply the angle scaled from [0, 180] to [0, 1]
        directional_distance = self.get_angle(ref_opinions) / np.pi

        attraction_mass = (1 + self_degree) / 2
        ref_attraction_mass = (1 + ref_degree) / 2

        # Attraction is based on a gravitation law. With the directional distance
        # reverse squared and the masses are the opinion centralisty attractiveness
        # The gravitanional constant is modulated by the emotion affectiveness
        attraction = math.exp(-(directional_distance ** 2) /
                              (g * attraction_mass * ref_attraction_mass))

        return attraction

    def rotate_opinions(self, 
                        ref_opinions: np.ndarray, 
                        angle: float
                        ) -> None:
        """
        Rotates the agent's opinion angle towards the reference by the given
        angle via Grandt-Schmidt orthogonalisation.

        Arguments:
            ref_opinions {np.ndarray} -- Reference opinions vector towards which
            the agent's opinion should rotate.
            angle {float} -- Angle of rotation.
        """
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

    def interact(self, ref_opinions: np.ndarray) -> None:
        """
        Performs the agent's opinion interaction with the external influence
        reference opinions. The rotation angle is calculated based on directional
        similarity and opinion strength.

        Arguments:
            ref_opinions {np.ndarray} -- The reference opinion vector that the
            agent interacts with.
        """

        # Check if there's any influence
        if np.any(ref_opinions):
            # Get the relative angle
            angle = self.get_angle(ref_opinions)
            # The relative strength of opinions dictate the bias of opinion rotation
            reference_strength = np.linalg.norm(
                ref_opinions) / np.sqrt(len(ref_opinions))
            own_strength = np.linalg.norm(
                self.opinions) / np.sqrt(len(ref_opinions))
            relative_strength = np.clip(
                reference_strength / own_strength, 0.5, 2)
            dissonance_factor = np.sin(2 * angle) / 2
            rotation_angle = angle * dissonance_factor * relative_strength
            # And do the job
            if VERBOSITY & V_AGENT:
                print(
                    f'Interacting: angle = {np.degrees(angle):.2f}\tdissonance_factor = {dissonance_factor:.3f}\trelative_strength = {relative_strength:.3f}\trotation_angle = {np.degrees(rotation_angle):.2f}')
            self.rotate_opinions(ref_opinions, rotation_angle)

    def add_noise(self, noise: np.ndarray) -> None:
        """
        Adds the randomly calculated noise and clip to range.

        Arguments:
            noise {np.ndarray} -- Vector of calculated noise components
        """
        # Just add the noise, it was already attenuated by emotions in the model scope
        self.opinions += noise
        # Make sure we're still in range
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)
