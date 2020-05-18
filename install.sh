pip3 install numpy scipy opencv-python
pip3 install -U torch==1.4+cpu torchvision==0.5+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install cython pyyaml==5.1
pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && python3 -m pip install -e .
cd projects/DensePose && wget -nc https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_WC1_s1x/173862049/model_final_289019.pkl
