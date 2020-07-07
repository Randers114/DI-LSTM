from helpers.helper_config import Config
from helpers.helper_enums import TrainingLoop
from layers.layer_mlp_gate import MLPGate
from models.model_controllers.model_controller import ModelController
from models.model_controllers.model_controller_custom import \
    ModelControllerCustomTraining
from models.model_controllers.model_controller_mlp import ModelControllerMLP
from models.model_novel import NovelModel

def get_model_controller():
    if (Config.training_loop_type == TrainingLoop.BaseFit):
        return ModelController(NovelModel())
    elif (Config.training_loop_type == TrainingLoop.MLP):
        return ModelControllerMLP(MLPGate())
    else:
        return ModelControllerCustomTraining(NovelModel())
