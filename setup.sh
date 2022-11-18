pip install argparse easydict h5py matplotlib numpy opencv-python pyyaml \
  scipy tensorboardX tqdm transforms3d typing-extensions
pip install open3d==0.10 timm==0.4.5

pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu116

HOME=`pwd`/pointr_detect/Lib

# Chamfer Distance
cd $HOME/chamfer_dist
python setup.py install --user

# NOTE: For GRNet

# Cubic Feature Sampling
cd $HOME/cubic_feature_sampling
python setup.py install --user

# Gridding & Gridding Reverse
cd $HOME/gridding
python setup.py install --user

# Gridding Loss
cd $HOME/gridding_loss
python setup.py install --user

pip install \
  "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade \
  https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

