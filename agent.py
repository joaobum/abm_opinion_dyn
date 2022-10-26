import numpy as np

from configuration import *

################################################################################
#       AGENT CLASS DEFINITION
################################################################################
class Agent:
    def __init__(self, id: int, init_orientation: float, init_emotion: float, init_adjacency: np.array) -> None:
        # Store needed values
        self.id = id
        self.emotion = init_emotion
        self.adjacency = init_adjacency
        
        
        self.opinions = []
        self.adjacency = []
        self.second_adjacency = []
        self.agents_trust = []
        self.media_trust = []
        self.media_conformity = []
        self.emotion = 0
        
        
        if VERBOSE:
            print(f"Initialising agent {id}:\torientation = {init_orientation:.3f}\temotion = {init_emotion:.3f}")
            
        # Store relevant data
        self.id = id
        self.orientation = init_orientation
        self.emotion = init_emotion
        # Then initialise our opinion
        self.initialise_opinions()
        
       
    def initialise_opinions(self, orientation):
        self.opinions = np.random.normal()
        
        
    def update_orientation(self, policies_orientations, orientations_normal):
        """
        Updates the agent's orientation based on its opinions.
        The agent's orientation is the root mean square of the agent's opinions multiplied by the orientation of each policy.
        Some transformations are required to maintain the sign on the square, but the base principle is that the agent's opinion
        on policies closer to neutral should have less effect on the agents opinion than its opinion in more tradical matters,
        in a quadratic increase.

        Args:
            policies_orientations (array of floats): The orientation of all the policies the agent will have an opinion on
        """
        mean_squared = np.mean([np.square(self.opinions[i] * policies_orientations[i])*np.sign(self.opinions[i] * policies_orientations[i]) for i in range(len(policies_orientations))])
        orientation = np.sqrt(np.abs(mean_squared)) * np.sign(mean_squared) / orientations_normal
        if VERBOSE:
            print(f"Updating agent {self.id} orientation:\t{self.orientation:.3f} -> {orientation:.3f}")
        self.orientation = orientation

               
    def get_angle(self, other_opinions):
        """
        Retrieves the angle between the agent's opinion array and the one in the other_opinons parameter.
        Angle calculated by the cosine of the dot product.

        Args:
            other_opinions (array of floats): The reference array of opinions for the agent's angle to be calculated on

        Returns:
            float: The angle in radians between the agent's opinion and the other_opinions parameter
        """
        cos = np.dot(self.opinions, other_opinions) / (np.linalg.norm(self.opinions) * np.linalg.norm(other_opinions))
        cos = np.clip(cos, -1, 1)
        return np.arccos(cos)
    
    
    def rotate_opinions(self, other_opinions, angle):
        """
        Rotates the agent's opinion angle towards the other_opinons parameter by the angle parameter amount.

        Args:
            other_opinions (array of floats): The reference array of opinions for the agent to rotate towards
            angle (float): The rotation angle in radians
        """
        if np.abs(angle) < 1e-5:
            if VERBOSE:
                print(f"Rotation angle is too small ({angle}), ignoring...")
            return
             
        # Gram-Schmidt orthogonalization
        versor_self = self.opinions / np.linalg.norm(self.opinions)
        axis = other_opinions - np.dot(versor_self, other_opinions) * versor_self
        versor_axis = axis / np.linalg.norm(axis)
        
        # Perform rotation
        I = np.identity(len(self.opinions))
        R = I + ( np.outer(versor_axis, versor_self) - np.outer(versor_self,versor_axis) ) * np.sin(angle) + ( np.outer(versor_self, versor_self) + np.outer(versor_axis,versor_axis) ) * (np.cos(angle)-1)
        self.opinions = np.matmul(R, self.opinions)
        
        # Make sure we're still in range
        self.opinions = np.clip(self.opinions, OPINION_MIN, OPINION_MAX)
        
        
    def interact(self, other_opinions):
        """
        Performs the agent's interaction with the other_opinons parameter.
        The rotation angle is calculaated by attenuating the totaal angle between the agents by 
        a dissonance factor ang the agent's emotional attachment.
        A rotation is there performed by the amount of the calculated rotation_angle towards other_opinions.

        Args:
            other_opinions (array of floats): The array of opinions for the agent to interact with.
        """
        # Get the relative angle
        angle = self.get_angle(other_opinions)
        # Now calculate the dissonance factor and use it to modulate the rotation angle together with agent's emotion
        dissonance_factor = np.sin(2 * angle) / 2
        rotation_angle = angle * dissonance_factor * (1 - self.emotion/2)
        # And do the job
        if VERBOSE:
            print(f"Interacting: angle = {np.degrees(angle):.2f}\tdissonance_factor = {dissonance_factor:.3f}\temotion = {self.emotion:.3f}\trotation_angle = {np.degrees(rotation_angle):.2f}")
        self.rotate_opinions(other_opinions, rotation_angle)
         
            
    def add_noise(self, noise_opinion, noise_intensity):
        """
        Adds a noise factor to the agent's opinion.
        The rotation angle is attenuated by the noise intensity and the agent's emotion.

        Args:
            noise_opinion (array of floats): An array of opinion values for the noise
            noise_intensity (float): The intensity factor of the noise to be applied
        """
        # Get the relative angle
        angle = self.get_angle(noise_opinion)
        # Modulate the roation angle with the noise intensity and the agent's emotion
        rotation_angle = angle * noise_intensity * (1 - self.emotion)
        # And do the job
        self.rotate_opinions(noise_opinion, rotation_angle)
        
        if VERBOSE:
            print(f"Adding noise: angle = {np.degrees(angle):.2f}\tnoise_intensity = {noise_intensity:.3f}\temotion = {self.emotion:.3f}\trotation_angle = {np.degrees(rotation_angle):.2f}")