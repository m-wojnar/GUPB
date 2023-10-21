from .controller import DeepLearnerController
from .controller_train import DeepLearnerControllerTrain

__all__ = [
    'POTENTIAL_CONTROLLERS'
]

POTENTIAL_CONTROLLERS = [
    DeepLearnerController('deep_learner')
]
