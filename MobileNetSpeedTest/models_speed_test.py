from tensorflow.keras import preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input
from time import time
from tensorflow import keras
from time import time
from PIL import Image
import onnxruntime
import numpy as np

def image_preprocessing(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))  # Resize to match the CIFAR-10 image size

    # Preprocess the image
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)/255.0
    
    return img_array


def keras_model_speed(model_path,img_array):
    MobileNet = keras.models.load_model(model_path)
    output=MobileNet.predict(img_array,verbose=0)
    list_times = []
    for i in range(100):
        time1 = time()
        output=MobileNet.predict(img_array,verbose=0)
        time2 = time()
        list_times.append(time2-time1)
    print(sum(list_times) / len(list_times))
        # Process the output
    predicted_class = np.argmax(output[0])
    print(f'Predicted class index: {predicted_class}')

def onnx_model_speed(model_path,img_array): #just change the model path to check quantized onnx model
    session = onnxruntime.InferenceSession(model_path)
    # Prepare the input data
    input_name = session.get_inputs()[0].name
    feed_dict = {input_name: img_array}

    # Run inference
    list_times = []
    for i in range(100):
        time1 = time()
        output = session.run(None, feed_dict)
        time2 = time()
        list_times.append(time2-time1)

    print(sum(list_times) / len(list_times))

    # Process the output
    predicted_class = np.argmax(output[0])
    print(f'Predicted class index: {predicted_class}')

def onnx_model_accuracy_check(model_path,x_test,y_test):
    session = onnxruntime.InferenceSession(model_path)
    # Prepare the input data
    input_name = session.get_inputs()[0].name
    x_predictions = np.array([])
    x_test = tf.keras.applications.resnet50.preprocess_input(x_test)/255.0
    for counter,image in enumerate(x_test):
        # print(image)
        image = np.expand_dims(image, axis=0)
        feed_dict = {input_name: image}

        # Run inference
        output = session.run(None, feed_dict)
        # print(counter+1)
        # Process the output
        predicted_class = np.argmax(output[0])
        x_predictions = np.append(x_predictions,[predicted_class])
    false_counter = 0
    for index,prediction in enumerate(x_predictions):
        if prediction != y_test[index]:
            false_counter += 1

    acc = 100 - ((false_counter/10000) * 100)
    print(acc)
    return acc , false_counter,x_predictions

(_,_),(x_test,y_test) = keras.datasets.cifar10.load_data()
acc = onnx_model_accuracy_check("/home/mohammad/Desktop/CIFAR10_FS/Gitlab/Report/MobileNetSpeedTest/models/your_model.onnx",x_test,y_test)
print(acc)