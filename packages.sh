sudo apt update
sudo apt install python3 idle3
sudo apt-get install python3-pip3
sudo apt-get install gfortran
sudo apt-get install libhdf5-dev libc-ares-dev libeigen3-dev
sudo apt-get install libatlas-base-dev libopenblas-dev libblas-dev
sudo apt-get install liblapack-dev
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install pybind11
sudo -H pip3 install Cython==0.29.21
sudo -H pip3 install h5py==2.10.0
pip3 install gdown
copy binairy
sudo cp ~/.local/bin/gdown /usr/local/bin/gdown
gdown https://drive.google.com/uc?id=1WDG8Rbi0ph0sQ6TtD3ZGJdIN_WAnugLO
sudo -H pip3 install tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl

pip3 install pandas
pip3 install -U numpy
pip3 install seaborn
pip3 install sklearn

pip3 install opencv-python
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
sudo apt-get install libqtgui4
sudo apt-get install python3-pyqt5
sudo apt-get install libqt4-test