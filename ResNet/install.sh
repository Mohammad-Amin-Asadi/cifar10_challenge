sudo apt-get install python3-pip
sudo apt -y install python3-venv
sudo python3 -m venv resnet_classifier_env
sudo source resnet_classifier_env/bin/activate
sudo pip install --timeout 1200 -i http://pypi.partdp.ir/root/pypi/+simple/ tensorflow==2.3.0 tensorflow-gpu==2.3.0 opencv-python tqdm matplotlib --trusted-host pypi.partdp.ir

