
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten

from helpers.helper_config import Config
from models.model_controllers.model_controller_regression import ModelControllerRegression
from models.model_loss import Custom_Metric
from models.model_validation_output import print_model_validation

class RegressionModel():
    def init_model(self):
        main_input = Input(shape=(3887), dtype="float32", name='main_input')
        composed_layer = Dense(Config.output_size, activation='softmax')(main_input)

        self.model = tf.keras.Model(inputs=[main_input], outputs=composed_layer)
        
        self.model.summary()

    def compile_self(self):
        self.init_model()
        self.model.compile(
            loss=Config.loss_function,
            optimizer=Config.optimizer,
            metrics=[Custom_Metric.correlation, Custom_Metric.intersection])
        return self.model

if __name__ == "__main__":
    controller = ModelControllerRegression(RegressionModel())

    controller.run_model("linear_regression_baseline")
    print_model_validation("linear_regression_baseline")
