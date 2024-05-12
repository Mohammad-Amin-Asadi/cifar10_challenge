from keras.src.optimizers.schedules import learning_rate_schedule
from os import getcwd, path
from numpy import array, random
import tensorflow as tf
from keras import Model
from keras.datasets import cifar10
from keras.layers import Add, GlobalAveragePooling2D, Dense, Flatten, Conv2D, Lambda, Input, BatchNormalization, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.initializers import HeNormal
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
import json

with open('models_config.json') as config_file:
    preconfig = json.load(config_file)["resnet"]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


class PreProcessingData():
    def __init__(self, preconfig):
        self.preconfig = preconfig
        self.config = self.model_configuration()

    def model_configuration(self):
        width, height, channels = x_train.shape[1:]
        print(channels)
        train_size = (1 - self.preconfig["validation_split"])*len(x_train)
        val_size = self.preconfig["validation_split"] * len(x_train)
        steps_per_epoch = tf.math.floor(
            train_size/self.preconfig["batch_size"])
        val_steps_per_epoch = tf.math.floor(
            train_size/self.preconfig["batch_size"])
        epochs = tf.cast(tf.math.floor(self.preconfig["max_iter_num"] / steps_per_epoch),
                         dtype=tf.int64)
        loss = CategoricalCrossentropy(from_logits=True)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.preconfig["iteration_boundaries"], self.preconfig["lr_values"])
        optimizer = SGD(learning_rate=lr_schedule,
                        momentum=self.preconfig["SGD_momentum"])
        tensorboard = TensorBoard(log_dir=path.join(
            getcwd(), "logs"), histogram_freq=1, write_images=True)
        checkpoint = ModelCheckpoint(
            path.join(getcwd(), "model_checkpoint"), save_freq="epoch")
        callbacks = [tensorboard, checkpoint]
        config_second = {
            "width": width,
            "height": height,
            "channels": channels,
            "steps_per_epoch": steps_per_epoch,
            "val_steps_per_epoch": val_steps_per_epoch,
            "epochs": epochs,
            "loss": loss,
            "lr_schedule": lr_schedule,
            "optimizer": optimizer,
            "tensorboard": tensorboard,
            "checkpoint": checkpoint,
            "callbacks": callbacks
        }
        print(steps_per_epoch)
        final_config = self.preconfig | config_second
        self.final_config = final_config
        return final_config

    def random_crop(self, img, crop_size):
        assert img.shape[2] == 3, 'Image must have 3 channels.'
        y, x = [random.randint(0, img.shape[i] - crop_size[i] + 1)
                for i in [0, 1]]
        return img[y:y+crop_size[0], x:x+crop_size[1], :]

    def crop_generator(self, batches, crop_length):
        while True:
            batch_x, batch_y = next(batches)
            batch_crops = array(
                [self.random_crop(image, (crop_length, crop_length)) for image in batch_x])
            yield batch_crops, batch_y

    def preprocessed_data(self):
        global x_train, y_train, x_test, y_test
        paddings = tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])
        x_train = tf.pad(x_train, paddings, mode="CONSTANT")
        y_train = tf.keras.utils.to_categorical(
            y_train, self.preconfig["num_classes"])
        y_test = tf.keras.utils.to_categorical(
            y_test, self.preconfig["num_classes"])
        train_generator = ImageDataGenerator(
            validation_split=self.final_config["validation_split"], rescale=1./255, horizontal_flip=True, preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
        train_batches = train_generator.flow(
            x_train, y_train, self.final_config["batch_size"], subset="training")
        validation_batches = train_generator.flow(
            x_train, y_train, self.final_config["batch_size"], subset="validation")
        train_batches = self.crop_generator(
            train_batches, self.config["height"])
        validation_batches = self.crop_generator(
            validation_batches, self.config["height"])
        test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                            rescale=1./255)
        test_batches = test_generator.flow(
            x_test, y_test, batch_size=self.final_config["batch_size"])
        return train_batches, validation_batches, test_batches, self.final_config


class ResNet():
    def __init__(self, config):
        self.config = config

    def residual_block(self, x, number_of_filters, match_filter_size=False):
        initializer = self.config["initializer"]
        x_skip = x
        if match_filter_size:
            x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(
                2, 2), kernel_initializer=self.config["initializer"], padding='same')(x_skip)
        else:
            x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(
                1, 1), kernel_initializer=self.config["initializer"], padding='same')(x_skip)
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Conv2D(number_of_filters, kernel_size=(3, 3), kernel_initializer=self.config["initializer"],
                   padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        if match_filter_size and self.config["shortcut_type"] == "identity":
            x_skip = Lambda(lambda x: tf.pad(x[:, ::2, ::2, :],
                                             tf.constant([[0, 0,], [0, 0], [0, 0],
                                                          [number_of_filters//4, number_of_filters//4]]),
                                             mode="CONSTANT"))(x_skip)
        elif match_filter_size and self.config["shortcut_type"] == "projection":
            x_skip = Conv2D(number_of_filters, kernel_size=(1, 1),
                            kernel_initializer=initializer, strides=(2, 2))(x_skip)
        x = Add()([x, x_skip])
        x = Activation("relu")(x)
        return x

    def residual_blocks(self, x):
        filter_size = self.config["initial_num_feature_maps"]
        for layer_group in range(3):
            for block in range(self.config["n_stacked_boxes"]):
                if layer_group > 0 and block == 0:
                    filter_size *= 2
                    x = self.residual_block(
                        x, filter_size, match_filter_size=True)
                else:
                    x = self.residual_block(x, filter_size)
        return x

    def model_base(self, shp):
        """
            Base structure of the model, with residual blocks
            attached.
        """
        # Get number of classes from model configuration
        initializer = self.config["initializer"]

        # Define model structure
        # logits are returned because Softmax is pushed  loss function.
        inputs = Input(shape=shp)
        x = Conv2D(self.config.get("initial_num_feature_maps"), kernel_size=(3, 3),
                   strides=(1, 1), kernel_initializer=initializer, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = self.residual_blocks(x)
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        outputs = Dense(self.config.get("num_classes"),
                        kernel_initializer=initializer)(x)

        return inputs, outputs

    def init_model(self):
        """
            Initialize a compiled ResNet model.
        """
        # Get shape from model configuration

        # Get model base
        inputs, outputs = self.model_base((32, 32, 3))

        # Initialize and compile model
        model = Model(inputs, outputs, name=self.config.get("name"))
        model.compile(loss=self.config.get("loss"),
                      optimizer=self.config.get("optimizer"),
                      metrics=self.config.get("optimizer_metric"))

        # Print model summary
        model.summary()

        return model

    def train_model(self, model, train_batches, validation_batches):
        tf.debugging.disable_traceback_filtering()
        model.fit(train_batches, batch_size=self.config["batch_size"], epochs=self.config["epochs"], verbose=self.config["verbose"],
                  callbacks=self.config["callbacks"],
                  steps_per_epoch=self.config["steps_per_epoch"],
                  validation_data=validation_batches,
                  validation_steps=self.config["val_steps_per_epoch"])
        self.model = model
        return model

    def evaluate_model(self, model, test_batches):
        score = model.evaluate(test_batches, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    def training_process(self):

        resnet = self.init_model()
        trained_resnet = self.train_model(
            resnet, train_batches, validation_batches)
        self.evaluate_model(trained_resnet, test_batches)
        return trained_resnet


if __name__ == "__main__":
    preprocessor_data = PreProcessingData(preconfig)
    train_batches, validation_batches, test_batches, final_config = preprocessor_data.preprocessed_data()
    resnet = ResNet(final_config)
    trained_resnet = resnet.training_process()
    tf.keras.models.save_model(trained_resnet, 'model7.keras')
