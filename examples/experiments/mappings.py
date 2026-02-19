# from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig
from experiments.a1x_pick_banana.config import TrainConfig as A1XDemoTrainConfig
from experiments.insert_block.config import TrainConfig as InsertBlockTrainConfig
from experiments.insert_hdmi.config import TrainConfig as InsertHDMITrainConfig
from experiments.insert_network_cable.config import TrainConfig as InsertNetworkCableTrainConfig
from experiments.fold_towel.config import TrainConfig as FoldTowelTrainConfig
from experiments.wipe_whiteboard.config import TrainConfig as WipeWhiteboardTrainConfig
from experiments.toast_bread.config import TrainConfig as ToastBreadTrainConfig
from experiments.pour_water.config import TrainConfig as pourwaterTrainConfig
from experiments.press_button.config import TrainConfig as pressbuttonTrainConfig


CONFIG_MAPPING = {
    # "task1_pick_banana": PickBananaTrainConfig,
    "a1x_pick_banana": A1XDemoTrainConfig,
    "insert_block": InsertBlockTrainConfig,
    "insert_hdmi": InsertHDMITrainConfig,
    "insert_network_cable": InsertNetworkCableTrainConfig,
    "fold_towel": FoldTowelTrainConfig,
    "wipe_whiteboard": WipeWhiteboardTrainConfig,
    "toast_bread": ToastBreadTrainConfig,
    "pour_water": pourwaterTrainConfig,
    "press_button": pressbuttonTrainConfig,
}
