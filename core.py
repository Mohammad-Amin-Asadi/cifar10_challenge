import argparse
from os import system
# argparse
parser = argparse.ArgumentParser(description='Process some aurguments')
parser.add_argument('--model_name', default='resnet', type=str, help='Any integer number')
args = parser.parse_args()
print(args.model_name)


model_name = str(args.model_name).lower() 
if model_name == "resnet":
    system(" ./resnet_classifier_env/bin/python ./ResNet/main.py")
elif model_name == "mobilenet":
    system(" ./resnet_classifier_env/bin/python ./MobileNet/main.py")
else:
    print("Wrong input")
