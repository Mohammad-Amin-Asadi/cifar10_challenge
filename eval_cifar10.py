import argparse
from keras.models import load_model
from keras.datasets import cifar10
# argparse
parser = argparse.ArgumentParser(description='Process some aurguments')
parser.add_argument('--model_dir', default='', type=str, help='Any integer number')
args = parser.parse_args()

# loading data
model =  load_model(args.model_dir)
print('preparing dataset ... ')
(_,_),(x_test,y_test) = cifar10.load_data()

# evaluate
print('evaluating ...')
score = model.evaluate(x_test,y_test,verbose=2)
print(f"Test loss: {score[0]} , Test Accuracy: {score[1]}")