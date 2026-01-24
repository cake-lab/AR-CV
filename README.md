
# Teleoperation using a CV Integrated System (ROS 2 Humble)

This repository contains a single ROS 2 package providing a vision-based teleoperation pipeline for the PiPER Arm.

## Overview

These CV codes are for the teleoperation and movement of the PiPER Arm using the CV+IMU Setup:

## Prerequisites

  * Ubuntu 22.04 LTS
  * ROS 2 Humble Hawksbill
    ```bash
    sudo apt update
    sudo apt install ros-humble-desktop
    ```
  * Python Virtual Environment
  * Python 3.10
  * CV2 with CUDA

## Installation
The installation of the repository can be done using the command prompt as follows:

1. **Open your Python Virtual Environment (in our case venv) and run the following commands**
   ```bash
   echo "alias sr='source venv/bin/activate'" >> ~/.bashrc
   source ~/venv/bin/activate # or simply run sr
   ```

   ```bash
   sudo apt update
   sudo apt install -y git cmake build-essential libusb-1.0-0-dev \ 
      pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \ 
      python3.10 python3.10-venv python3.10-dev
   ```
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install "numpy<2.0"
   ```
   ```bash
   pip pip show numpy # To check if the numpy version is < 2.0
   ```

2. **Compile PyRealSense 2.50.0 from Source**
   ```bash
   cd ~
   git clone https://github.com/IntelRealSense/librealsense.git
   cd librealsense
   git checkout v2.50.0 
   ```
  **Build the Python Environment**
   ```bash
   mkdir build && cd build

   cmake .. \
   -DCMAKE_BUILD_TYPE=Release \
   -DBUILD_PYTHON_BINDINGS=bool:true \
   -DPYTHON_EXECUTABLE=$(which python3) \
   -DBUILD_EXAMPLES=false \
   -DBUILD_GRAPHICAL_EXAMPLES=false
   ```
   **Compile the Python Environement**
   ```bash
   make -j$(nproc)
   ``` 
   **Install the PyRealSense into your virtual environment**
   ```bash
   find . -name "pyrealsense2*.so" # Locate the compiled .so file
   cp ./wrappers/python/pyrealsense2*.so ~/venv/lib/python3.10/site-packages/ # Copy it to your venv (Adjust the path if your find command output differs)
   ```
   * Test the PyRealSense driver in your virtual environment (venv)
   ```bash
   python -c "import pyrealsense2 as rs; print(f'RealSense Version: {rs.__version__}')"
   ```
   **Install PyTorch and other Dependencies**
   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 # PyTorch for CUDA 11.6
   pip install scipy pyserial mediapipe websockets
   ```

3. **Compile OpenCV and CUDA**
  **Configure the CMake Environment**
   ```bash
   cd ~/opencv/build
   rm -rf *

   cmake -D CMAKE_BUILD_TYPE=RELEASE \
   -D CMAKE_INSTALL_PREFIX=$(python3.10 -c "import sys; print(sys.prefix)") \
   -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
   -D WITH_CUDA=ON \
   -D WITH_CUDNN=OFF \
   -D OPENCV_DNN_CUDA=OFF \
   -D CUDA_ARCH_BIN=7.5 \
   -D WITH_CUBLAS=1 \
   -D ENABLE_FAST_MATH=1 \
   -D WITH_IMAGEIO=ON \
   -D BUILD_opencv_python3=ON \
   -D PYTHON3_EXECUTABLE=$(which python3.10) \
   -D PYTHON3_INCLUDE_DIR=$(python3.10 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
   -D PYTHON3_PACKAGES_PATH=$(python3.10 -c "import site; print(site.getsitepackages()[0])") \
   -D BUILD_EXAMPLES=OFF \
   -D BUILD_TESTS=OFF \
   ..
   ```
   **Compile the Python Environement**
   ```bash
   make -j$(nproc)
   ``` 
   **Compile the Python Environement**
   ```bash
   make install
   ``` 

3. **Check if dependencies are installed (recommended)**
   ```bash
   source ~/venv/bin/activate # or simply run sr 
   
   #Add CV2 to venv
   cd ~/venv/lib/python3.10/site-packages/cv2/python-3.10
   mv cv2.cpython-310-x86_64-linux-gnu.so ../../
   
   # Verify PyRealSense presence
   python -c "import pyrealsense2 as rs; print('SUCCESS: RealSense found')"
   '''
   * If PyRealSense is not detected, run the below commands again:
   ```bash
   cd ~/librealsense/build
   find . -name "pyrealsense2*.so"
   cp ./wrappers/python/pyrealsense2*.so ~/venv/lib/python3.10/site-packages/
   ``` 

4. **Run your Scripts**
   ```bash
   cd "~/Desktop/Cake Lab/CV"
   python final.py # To run the CV+IMU Setup
   
   # Optional 
   # aruco2.py - used to provide user feedback for the ARCore App
   # only use for visual check of the shape tracing not needed for running the CV+IMU Setup 
   python aruco2.py 
   ``` 

## Running the CV+IMU Setup

Run the final.py in your terminal

```bash
source ~/venv/bin/activate # or simply run sr
cd "~/Desktop/Cake Lab/CV"
   python final.py # To run the CV+IMU Setup
```

| Keys   | Purpose                                          |
| :----- | :----------------------------------------------- |
| `1-5`  | Render Shapes                                    |
| `Q`    | Killing Terminal                                 |
| `W`    | Changing Tracking Mode                           |
| `H`    | Changing Hand (to select Right(default) or Left) |
| `F`    | Switching between filters (Euro and EMA)         |


Feel free to open issues or submit pull requests for improvements\!

