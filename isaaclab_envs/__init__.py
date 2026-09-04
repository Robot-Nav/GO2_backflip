"""IsaacLab 版 GO2 任务注册表（与原 ``tasks`` 包结构对齐）。"""

from .go2_backflip import Go2Backflip
from .go2_sideflip import Go2Sideflip
from .go2_sideroll import Go2Sideroll
from .go2_twohand import Go2Twohand

task_dict = {
    "Go2Backflip": Go2Backflip,
    "Go2Sideflip": Go2Sideflip,
    "Go2Sideroll": Go2Sideroll,
    "Go2Twohand": Go2Twohand,
}
