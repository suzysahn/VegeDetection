# VegeDetection
## Under construction

Reccomended thing to install on your laptop before running to make it easier to copy/paste files from desktop to the pi: https://pimylifeup.com/raspberry-pi-samba/ (name your folder shared inside of home on pi to have correct path later on)

Packages to download 
- TensorFlow 2.4.0.rc (check with "python3 -c 'import tensorflow as tf; print(tf.__version__)'") follow: https://qengineering.eu/install-tensorflow-2.4.0-on-raspberry-64-os.html
- numpy
- python 3 (make sure this is python3!!, check with python3 --version)
- any other packages it needs should be installed with pip3 (not pip since it's gonna want to download with python 2)
- libjasper/other libatlas packages https://github.com/amymcgovern/pyparrot/issues/34
- sudo apt install libqt4-test

if training again
- pip3 install pandas
- pip3 install -U numpy
- pip3 install seaborn
pip3 install sklearn

Before adding on the camera module to the system, test model with ModelWOCam.py