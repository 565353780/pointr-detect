pip install argparse easydict h5py matplotlib numpy opencv-python pyyaml \
  scipy tensorboardX tqdm transforms3d typing-extensions numpy-quaternion
pip install open3d==0.10 timm==0.4.5

pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install \
  "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade \
  https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

cd ./point_shape_detect/Lib/chamfer_dist
python setup.py install --user

