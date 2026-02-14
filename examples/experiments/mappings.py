# from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig
from experiments.a1x_pick_banana.config import TrainConfig as A1XDemoTrainConfig
from experiments.Insert_block.config import TrainConfig as InsertBlockTrainConfig
from experiments.Insert_HDMI.config import TrainConfig as InsertHDMITrainConfig
from experiments.Insert_network_cable.config import TrainConfig as InsertNetworkCableTrainConfig
from experiments.Fold_towel.config import TrainConfig as FoldTowelTrainConfig
from experiments.Wipe_whiteboard.config import TrainConfig as WipeWhiteboardTrainConfig
from experiments.Toast_bread.config import TrainConfig as ToastBreadTrainConfig



CONFIG_MAPPING = {
    # "task1_pick_banana": PickBananaTrainConfig,
    "a1x_pick_banana": A1XDemoTrainConfig,
    "insert_block": InsertBlockTrainConfig,
    "insert_hdmi": InsertHDMITrainConfig,
    "insert_network_cable": InsertNetworkCableTrainConfig,
    "fold_towel": FoldTowelTrainConfig,
    "wipe_whiteboard": WipeWhiteboardTrainConfig,
    "toast_bread": ToastBreadTrainConfig,
}
