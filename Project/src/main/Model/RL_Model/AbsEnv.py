import abc as absract
import gymnasium as gym
from typing import Tuple, Any, Dict, Optional

class AbsEnv(gym.Env, absract.ABC):
    @absract.abstractmethod
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, 
              **kwargs) -> Tuple:
        pass

    @absract.abstractmethod
    def step(self, action: object) -> Tuple:
        pass

    @absract.abstractmethod
    def render(self) -> None:
        pass

    @absract.abstractmethod 
    def close(self) -> None:
        pass