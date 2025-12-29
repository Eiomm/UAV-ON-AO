from enum import Enum
from typing import Dict
from common.param import args


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )
        return cls._instances[cls]


# ====== 动作 ID（env_uav.py 需要的）======
class AirsimActions(Enum):
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4


# ====== 动作参数（步长 / 角度）======
class _DefaultAirsimActionSettings(Dict):
    FORWARD_STEP_SIZE = args.xOy_step_size
    UP_DOWN_STEP_SIZE = args.z_step_size
    LEFT_RIGHT_STEP_SIZE = args.xOy_step_size
    TURN_ANGLE = args.rotateAngle


AirsimActionSettings = _DefaultAirsimActionSettings()