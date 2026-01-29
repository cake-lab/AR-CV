**AR-CV: Vision Teleoperation System (CUDA Accelerated)**  

This repository contains a high-performance, GPU-accelerated teleoperation pipeline for the PiPER Arm using Intel RealSense and OpenCV.  
This guide builds a local virtual environment with:  
- OpenCV 4.x (Compiled from source with CUDA & GTK support)  
- Intel RealSense (Compiled with RSUSB backend for kernel stability)  
- PyTorch 2.x (GPU accelerated)  

**Prerequisites**
* **OS**: Ubuntu 22.04 LTS  
* **GPU**: NVIDIA GPU (RTX 20-series or newer recommended)  
* **Drivers**: NVIDIA Driver 535+ (Check with nvidia-smi)  
* **CUDA**: Version 11.8 or newer installed on system.  

**Critical: Choose Your Architecture**  
Before compiling OpenCV (Phase 3), you must know your GPU's Compute Capability. Use this table:  


| GPU Series                       | Architecture Flag (CUDA_ARCH_BIN) |  
|--------------------------------- |---------------------------------- |  
| RTX 40-series (4060, 4070, 4090) | 8.9                               |  
| RTX 30-series (3060, 3070, 3080) | 8.6                               |  
| RTX 20-series (2060, 2070, 2080) | 7.5                               |  
| GTX 10-series (1060, 1070, 1080) | 6.1                               |  
  
**Phase 1: System Prep & Skeleton**  
We start by cleaning the system and setting up the project structure.  
  
Install Build Tools (Use aptitude to resolve dependency conflicts automatically)  
```bash
# [SYSTEM TERMINAL]
sudo apt update
sudo apt install aptitude -y
sudo aptitude install git cmake build-essential libusb-1.0-0-dev \
   pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
   python3.10 python3.10-venv python3.10-dev libssl-dev
```
  
Clone & Create Venv  
```bash
# [SYSTEM TERMINAL]
cd ~
git clone [https://github.com/cake-lab/AR-CV.git](https://github.com/cake-lab/AR-CV.git) AR-CV
cd AR-CV
# Create virtual environment LOCALLY inside the project
python3 -m venv venv
# Activate and update base tools
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install "numpy<2.0" # Critical: Pin NumPy to 1.x for OpenCV compatibility
```
 
**Phase 2: Compile RealSense (RSUSB Backend)**  
We compile librealsense from source to force the RSUSB backend, which fixes "No Device Connected" errors on modern Linux kernels.  
  
Get Source & Permissions  
```bash
# [SYSTEM TERMINAL] - Do not run in venv
deactivate 2>/dev/null
cd ~
git clone [https://github.com/IntelRealSense/librealsense.git](https://github.com/IntelRealSense/librealsense.git)
cd librealsense
git checkout v2.50.0
# Install permissions (requires sudo)
sudo ./scripts/setup_udev_rules.sh
# ACTION: Unplug and Replug your camera now!
```
   
Build & Install  
```bash
# [SYSTEM TERMINAL]
mkdir build && cd build
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DBUILD_PYTHON_BINDINGS=bool:true \
-DPYTHON_EXECUTABLE=$(which python3) \
-DBUILD_EXAMPLES=false \
-DBUILD_GRAPHICAL_EXAMPLES=false \
-DFORCE_RSUSB_BACKEND=true
make -j$(nproc)
sudo make install
sudo ldconfig
```
 
Link to Project Venv  
```bash
# [SYSTEM TERMINAL]
# Copy the compiled library into our local venv
cp ~/librealsense/build/wrappers/python/pyrealsense2*.so ~/AR-CV/venv/lib/python3.10/site-packages/
# Rename it so Python can import it easily
mv ~/AR-CV/venv/lib/python3.10/site-packages/pyrealsense2*.so ~/AR-CV/venv/lib/python3.10/site-packages/pyrealsense2.so
```
   
**Phase 3: Compile OpenCV (CUDA + GTK)**  
This builds a custom OpenCV 4.x with NVIDIA CUDA acceleration and GTK (GUI) support.  
  
Download Source  
```bash
# [SYSTEM TERMINAL]
cd ~
git clone [https://github.com/opencv/opencv.git](https://github.com/opencv/opencv.git)
git clone [https://github.com/opencv/opencv_contrib.git](https://github.com/opencv/opencv_contrib.git)
```
 
Configure CMake  
**CRITICAL**: Update CUDA_ARCH_BIN below to match your GPU (See table in Prerequisites).  
```bash
# [IN VENV] - We MUST be inside venv to link Python correctly
cd ~/AR-CV
source venv/bin/activate
cd ~/opencv
mkdir -p build && cd build
rm -rf * # Clean start
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D WITH_CUDA=ON \
-D WITH_CUDNN=OFF \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_ARCH_BIN=8.9 \
-D WITH_CUBLAS=1 \
-D ENABLE_FAST_MATH=1 \
-D WITH_IMAGEIO=ON \
-D WITH_GTK=ON \
-D BUILD_JAVA=OFF \
-D BUILD_opencv_java_bindings_generator=OFF \
-D BUILD_opencv_python3=ON \
-D OPENCV_PYTHON3_INSTALL_PATH=~/AR-CV/venv/lib/python3.10/site-packages \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
..
```
   
Compile & Install  
```bash
# [IN VENV]
make -j$(nproc)
sudo make install
sudo ldconfig
```   
  
Final Cleanup (Linking)  
```bash
# [IN VENV]
# Manually copy the file to ensure it's in the right place
cp ~/opencv/build/lib/python3/cv2*.so ~/AR-CV/venv/lib/python3.10/site-packages/
# Rename for import
cd ~/AR-CV/venv/lib/python3.10/site-packages/
rm -f cv2.so # Remove any CPU-only fallback if it exists
mv cv2.cpython-310-x86_64-linux-gnu.so cv2.so
cd ~/AR-CV # Return home
```
   
**Phase 4: Final Dependencies**
Install the remaining AI logic libraries. We use PyTorch with CUDA 11.8 support (which is compatible with CUDA 12 drivers).  
```bash
# [IN VENV]
source venv/bin/activate
# 1. Install PyTorch (Stable GPU Version)
pip install torch==2.0.1 torchvision==0.15.2 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
# 2. Install Logic Libs (Pinned for stability)
# Note: We pin numpy<2.0 to prevent conflicts with OpenCV/RealSense
pip install "numpy<2.0" scipy pyserial websockets "mediapipe==0.10.14"
# 3. Clean up any CPU-OpenCV that MediaPipe might have auto-installed
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
```   
  
**Phase 5: Verification**  
Create a file named libcheck.py to verify your GPU stack is active.  
```bash
import torch
import cv2
import pyrealsense2 as rs
print("--- Library Check ---")
print(f"PyTorch: {torch.__version__}")
print(f"OpenCV:  {cv2.__version__}")
print(f"RealSense: {rs.__version__}")
if torch.cuda.is_available():
    print(f"SUCCESS: PyTorch GPU -> {torch.cuda.get_device_name(0)}")
else:
    print("FAIL: PyTorch cannot see GPU")
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    if count > 0:
        print(f"SUCCESS: OpenCV GPU -> Found {count} device(s)")
    else:
        print("FAIL: OpenCV compiled without CUDA")
except:
    print("FAIL: cv2.cuda module missing")
```  
   
Run the check:  
```bash
python libcheck.py
```

**Running the CV+IMU Setup**
Run the final.py in your terminal  

```bash
source ~/venv/bin/activate # or simply run sr
cd ~/Ar-CV
   python final.py # To run the CV+IMU Setup
```
| Keys |	Purpose                                          |
|----- |-------------------------------------------------- |
| 1-5  |	Render Shapes                                    |
| Q    |	Killing Terminal                                 |
| W    |	Changing Tracking Mode                           |
| H    |	Changing Hand (to select Right(default) or Left) |
| F    |	Switching between filters (Euro and EMA)         |  

Feel free to open issues or submit pull requests for improvements!
