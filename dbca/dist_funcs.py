

import numpy as np
from collections import Counter

from dbca.freq_distribution import FrequencyDistribution




def chernoff_similarity(P: FrequencyDistribution, Q: FrequencyDistribution, alpha: float = 0.5) -> float:
    """
    Return Chernoff similarity between 2 frequency distributions

    Parameters
    ----------
    P : FrequencyDistribution
    Q : FrequencyDistribution
    alpha: float
        Chernoff co-efficient (scalar in [0,1])

    Returns
    -------
    float
        Chernoff similarity.

    """
    all_elements = sorted(list(P.elements | Q.elements)) # ensure same iteration order
    p_probs = np.array([P.element_prob(e) for e in all_elements])**alpha
    q_probs = np.array([Q.element_prob(e) for e in all_elements])**(1-alpha)
    return np.dot(p_probs, q_probs)


def chernoff_divergence(P: FrequencyDistribution, Q: FrequencyDistribution, alpha: float = 0.5) -> float:
    """
    Return complement of Chernoff similarity between two distributions

    Parameters
    ----------
    P : FrequencyDistribution
    Q : FrequencyDistribution
    alpha : float, optional
        Chernoff co-efficient. The default is 0.5.

    Returns
    -------
    float
        DESCRIPTION.

    """
    return 1 - chernoff_similarity(P, Q, alpha)

