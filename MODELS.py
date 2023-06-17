"""
The code here is quite clunky. 

Sooner or later, I will rewrite it to be somewhat sklearn compatible, and delete redundant things. 
sklearn has neat naming conventions, a nice estimator base class, and parameter validation conventions. 

Will need to implement partial likelihood estsimation here as well. 
"""

import numpy as np
from typing import Tuple # 'iterable', see sklearn param_validation decorator

# Cython code to generate, and perform relevent computations, across permutations of items
import pyximport
pyximport.install()
import permute_jk


class TiedRankingLogitModel:
    """
    Link to paper:
    https://statisticalhorizons.com/wp-content/uploads/2022/01/AllisonChristakis.pdf
    
    Here, we list some assumptions about the use of this class.
    
    1) We assume that there are no 'empty' response fields.

    2) We also assume we are exponent-iating a common linear index function
     - with each index weight applied to a possible multinomial choice.
    Refer to 5.1 of the paper, in their provided example. 

    3) The parameters input should be an array, with each element corresponding to
    the linear index weight (signed). This ordering of categories is arbitrary, but should
    be the same as the observed_rankings input.
        - it will be good to provide, in comments, a mapping of the index value in the list
          to some category names. 
        - Refer to page 210 for this mu_ij form

    4) Existimg implementation for estimation: Q estimates it by pivoting on one of the outcomes.
    So, we assume that parameters[0] == 0 all the time. We also calculate the llhoods
    based on this pivoting characterization of a multinomial logit model.
    See: https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_set_of_independent_binary_regressions
    """
    def __init__(self, j: int, parameters: Tuple[float, ...]):
        # j is number of categories there exist
        self.valid_init = 0
        self.j = j
        self.parameters = parameters

    def check_valid_init(self):
        """
        Call before using this function object to ensure valid initialisation of the
        underlying statistical model (correct observation, parameters, so forth)
        """
        num_parameters_inputted = len(self.parameters)
        if self.j == num_parameters_inputted:
            self.valid_init = 1

    def evaluate_llhood(self, observed_ranking: Tuple[int, ...]):
        """
        Valid observation inputs are numeric upwards,
        to a maximum of j, each in the list index corresponding to an outcome specified
        in the parameters input.
        """
        # NOTE: Integer input currently based on Q CID output format
        # Highest integer corresponds to 'best (first)' ranked variable. 
        # Then each subsequent lower integer (does not have to decrease sequentially) is a lower level of preference.
        
        llhood: float = 1.0  # product of llhoods, so we start with 1

        if len(set(observed_ranking)) == 1: # This ensures we do not get stuck in an infinite loop below, as ensures there are two distinct ranks.
                                            # THIS IS NOT THE BEST WAY TO HANDLE IT - NEEDS FIX
            return llhood

        exponentiated_params = np.exp(np.asarray(self.parameters))

        enumerated = list(enumerate(observed_ranking))
        i = max(observed_ranking) # start with highest number in input tuple, and iterate down.
        while True:  # outer product
            tied_ranks_indexes = [index for index, number in enumerated if number == i]
            if len(tied_ranks_indexes) == 0:
                i -= 1
                continue # continue through, but keep checking for lower integers
            lower_ranks_indexes = [index for index, number in enumerated if number < i]
            if len(lower_ranks_indexes) == 0:
                break  # we are on the last ranking, and there is no llhood term
            
            tied_ranks_terms = np.asarray([exponentiated_params[i] for i in tied_ranks_indexes])
            lower_ranks_sum = np.sum([exponentiated_params[i] for i in lower_ranks_indexes])

            llhood *= permute_jk.permuteexpression(tied_ranks_terms, lower_ranks_sum)
            i -= 1
        
        return llhood


class LatentClassSpecificWrapperModel:
    """
    Here, you can wrap up the underlying statistical models used in each of the latent classes
    into one.

    Observations should be a tuple of observation inputs, which are valid inputs for the evaluate_llhood
    functions of the individual models. The number of items in both arguments must be the same.
    """
    def __init__(self, models: Tuple):
        self.models = models

    def evaluate_llhood(self, observations: Tuple):  # observations should be a tuple of valid observations
        llhood: float = 1.0
        for i in range(len(self.models)):
            llhood *= self.models[i].evaluate_llhood(observations[i])
        return llhood


class LCAMixtureModel:
    """
    Now for a linearly weighted Bayesian mixture model, you can evaluate posterior probabilities
    """
    def __init__(self, classes: int, models: Tuple, linear_weights: Tuple[float, ...]):
        self.num_classes = classes
        self.models = models
        self.priors = linear_weights

    def evaluate_posterior_probability(self, observation):
        # Ensure observation format matches the class specific model in the mixture model framework
        weighted_llhoods: list[float] = list(range(self.num_classes))
        for i in range(self.num_classes):
            weighted_llhoods[i] = self.priors[i] * self.models[i].evaluate_llhood(observation)
        denom = sum(weighted_llhoods)

        output: list[float] = []
        for i in range(len(weighted_llhoods)):
            output[i] = weighted_llhoods[i]/denom

        return output

    def label_observation(self, observation):
        # Ensure observation format matches the class specific model in the mixture model framework
        weighted_llhoods: list[float] = list(range(self.num_classes))
        for i in range(self.num_classes):
            weighted_llhoods[i] = self.priors[i] * self.models[i].evaluate_llhood(observation)
        denom = sum(weighted_llhoods)

        output: list[float] = list(range(self.num_classes))
        for i in range(self.num_classes):
            output[i] = weighted_llhoods[i] / denom

        return np.argmax(output) + 1
    
    
class GeneralMultinoulliModel:  # multinomial but n=1, hence 'noulli'
    """
    The observation integer should correspond to the index of the probabilities
    you provide. 
    """
    def __init__(self, k: int, probabilities: Tuple[float, ...]):  # model parameters
        # e.g. k=2 binomial, k=3 trinomial...
        # in the probability parameter vector, you NEED to specify
        # one explicit probability for each outcome
        self.k = k
        self.probabilities = probabilities
        self.valid_init = 0
        
    def check_valid_init(self):
        """
        Call before using this function object to ensure valid initialisation of the
        underlying statistical model (correct observation, parameters, so forth)
        """
        num_probabilities_inputted = len(self.probabilities)
        if num_probabilities_inputted == self.k and sum(self.probabilities) == 1:
            self.valid_init = 1

    def evaluate_llhood(self, observation: int):
        """
        Returns None is the observation is not a valid input
        """
        if observation not in list(range(len(self.probabilities))):
            return None
        llhood = self.probabilities[int(observation)]
        return llhood
    
