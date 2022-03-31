# ros_pvrcnn_detectwithvison
use open-mmlab/OpenPCDet pvrcnn to detecte for autodrive
May be able also to use pointpillars、second...

1 Openpcdet
https://github.com/open-mmlab/OpenPCDet
2 make
mkdir workspace
cd workspace
mkdir src
cd src
git clone https://github.com/xinsidao/ros_pvrcnn_detectwithvison.git
cd ..
catkin_make
3 run
dowmload pth to lidar_detect/src/ckpt
copy cfg/xxxx.cfg to lidar_detect/src/ckpt
source devel/setup.bash
python src/lidar_detect/src/getpointcloudep16.py

source devel/setup.bash
python src/lidar_detect/src/pubcarstop.py

