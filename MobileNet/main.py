from keras.src.optimizers.schedules import learning_rate_schedule
import os
import numpy as np
import tensorflow as tf
from keras.layers.experimental.preprocessing import Resizing
from keras import Model
from keras.datasets import cifar10
from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, Input, BatchNormalization,ReLU,DepthwiseConv2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.initializers import HeNormal
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
import json

with open('models_config.json') as config_file:
    preconfig = json.load(config_file)["mobilenet"]


(x_train,y_train),(x_test,y_test) = cifar10.load_data()
# Resizing(32,32, interpolation="bilinear", input_shape=x_train.shape[1:])

class PreProcessing_data():
    def __init__(self,preconfig):
        self.preconfig = preconfig
        self.config = self.model_configuration()

    def model_configuration(self):
        width,height,channels = x_train.shape[1:]
        print(channels)
        train_size = (1 - self.preconfig["validation_split"])*len(x_train)
        val_size = self.preconfig["validation_split"] * len(x_train)
        steps_per_epoch = tf.math.floor(train_size/self.preconfig["batch_size"])
        val_steps_per_epoch = tf.math.floor(train_size/self.preconfig["batch_size"])
        epochs = tf.cast(tf.math.floor(self.preconfig["max_iter_num"] / steps_per_epoch),\
		    dtype=tf.int64)
        loss = CategoricalCrossentropy(from_logits=True)
        lr_schedule  =tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.preconfig["iteration_boundaries"],self.preconfig["lr_values"])
        optimizer = SGD(learning_rate=lr_schedule,momentum=self.preconfig["SGD_momentum"])
        tensorboard = TensorBoard(log_dir=os.path.join(os.getcwd(),"log6"),histogram_freq=1)
        checkpoint = ModelCheckpoint(os.path.join(os.getcwd(), "model_checkpoint6"), save_freq="epoch")
        callbacks = [tensorboard,checkpoint]
        config_second = {
            "width":width,
            "height":height,
            "channels":channels,
            "steps_per_epoch":steps_per_epoch,
            "val_steps_per_epoch":val_steps_per_epoch,
            "epochs":95,
            "loss":loss,
            "lr_schedule":lr_schedule,
            "optimizer":optimizer,
            "tensorboard":tensorboard,
            "checkpoint":checkpoint,
            "callbacks":callbacks
        }
        print(steps_per_epoch)
        final_config  = self.preconfig | config_second
        self.final_config = final_config
        return final_config

    def random_crop(self, img, crop_size):
        assert img.shape[2] == 3,'Image must have 3 channels.'
        y, x = [np.random.randint(0, img.shape[i] - crop_size[i] + 1) for i in [0, 1]]
        return img[y:y+crop_size[0], x:x+crop_size[1], :]

    def crop_generator(self, batches, crop_length):
        while True:
            batch_x, batch_y = next(batches)
            batch_crops = np.array([self.random_crop(image, (crop_length, crop_length)) for image in batch_x])
            yield batch_crops, batch_y

    def preprocessed_data(self):
        global x_train , y_train,x_test,y_test
        paddings = tf.constant([[0,0],[4,4],[4,4],[0,0]])
        x_train = tf.pad(x_train,paddings,mode="CONSTANT")
        y_train = tf.keras.utils.to_categorical(y_train,self.preconfig["num_classes"])
        y_test = tf.keras.utils.to_categorical(y_test,self.preconfig["num_classes"])
        train_generator = ImageDataGenerator(validation_split=self.final_config["validation_split"],rescale=1./255,horizontal_flip=True,preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
        train_batches = train_generator.flow(x_train,y_train,self.final_config["batch_size"], subset="training")
        validation_batches = train_generator.flow(x_train,y_train,self.final_config["batch_size"], subset="validation")
        train_batches = self.crop_generator(train_batches, self.config["height"])
        validation_batches = self.crop_generator(validation_batches, self.config["height"])
        test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                            rescale=1./255)
        test_batches = test_generator.flow(x_test,y_test,batch_size=self.final_config["batch_size"])
        return train_batches,validation_batches,test_batches ,self.final_config



class MobilenetV1():
    def __init__(self,config):
      self.config = config


    def depth_block(self,x, strides):
        x = DepthwiseConv2D(3,strides=strides,padding='same',  use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def single_conv_block(self,x,filters):
        x = Conv2D(filters, 1,use_bias=False)(x)
        x= BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def combo_layer(self,x,filters,strides):
        x = self.depth_block(x,strides)
        x = self.single_conv_block(x, filters)
        return x
    def init_model(self):
        input = Input ((32,32,3))
        x = Conv2D(32,3,strides=(1,1),padding = 'same', use_bias=False) (input)
        x =  BatchNormalization()(x)
        x = ReLU()(x)
        x = self.combo_layer(x,64, strides=(1,1))
        x = self.combo_layer(x,128,strides=(1,1))
        x = self.combo_layer(x,128,strides=(2,2))
        x = self.combo_layer(x,256,strides=(1,1))
        x = self.combo_layer(x,256,strides=(1,1))
        x = self.combo_layer(x,512,strides=(2,2))
        for _ in range(5):
          x = self.combo_layer(x,512,strides=(1,1))
        x = self.combo_layer(x,512,strides=(2,2))
        x = self.combo_layer(x,1024,strides=(1,1))
        x = self.combo_layer(x,1024,strides=(1,1))
        x = GlobalAveragePooling2D()(x)
        output = Dense(self.config["num_classes"],activation='softmax')(x)
        model = Model(input, output)
        model.compile(loss=self.config.get("loss"),\
            optimizer=self.config.get("optimizer"),\
                metrics=self.config.get("optimizer_metric"))
        model.summary()
        return model
    def train_model(self,model,train_batches,validation_batches):
        tf.debugging.disable_traceback_filtering()
        model.fit(train_batches,batch_size=self.config["batch_size"],epochs=self.config["epochs"], verbose=self.config["verbose"],
                  callbacks=self.config["callbacks"],
                  steps_per_epoch=self.config["steps_per_epoch"],
                  validation_data=validation_batches,
                  validation_steps=self.config["val_steps_per_epoch"])
        self.model = model
        return model

    def evaluate_model(self,model,test_batches):
        score = model.evaluate(test_batches,verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    def training_process(self):

        resnet = self.init_model()
        trained_resnet = self.train_model(resnet,train_batches,validation_batches)
        self.evaluate_model(trained_resnet,test_batches)
        return trained_resnet

if __name__ == "__main__":
    preprocessor_data = PreProcessing_data(preconfig)
    train_batches,validation_batches,test_batches,final_config = preprocessor_data.preprocessed_data()
    mobile_net_model = MobilenetV1(final_config)
    trained_resnet = mobile_net_model.training_process()
    tf.keras.models.save_model(trained_resnet,'MobileNetV1.keras')