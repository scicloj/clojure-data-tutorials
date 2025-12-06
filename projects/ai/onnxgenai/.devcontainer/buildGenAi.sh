#!/bin/bash -ex
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb 
# sudo dpkg -i cuda-keyring_1.1-1_all.deb  
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin 
# sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 
# sudo apt-get update 
# sudo apt-get install -y cuda-toolkit-12 cudnn9-cuda-12
apt-get update
apt-get install -y cmake python3.12-dev python3.12-venv  build-essential
cd /root
git clone https://github.com/microsoft/onnxruntime-genai
cd /root/onnxruntime-genai
git checkout v0.11.2
python3 -m venv .venv 
source .venv/bin/activate
pip install requests
export CUDA_HOME=/usr/local/cuda
python build.py --use_cuda --build_java --config Release --publish_java_maven_local
cp /root/onnxruntime-genai/src/java/build/libs/onnxruntime-genai-0.11.2.jar /tmp/
mkdir /tmp/onnxruntime-genai.native
cp /root/onnxruntime-genai/build/Linux/Release/*.so /tmp/onnxruntime-genai.native
cp /root/onnxruntime-genai/build/Linux/Release/src/java/libonnxruntime-genai-jni.so /tmp/onnxruntime-genai.native/