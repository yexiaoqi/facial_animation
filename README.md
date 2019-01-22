# facial_animation
直接3d特征点，不进行向2d的投影
由于'Data/net-data/256_256_resfcn256_weight.data-00000-of-00001'和'shape_predictor_68_face_landmarks.dat'两个文件超过了github大小限制，保存在了本地未上传

首先要添加依赖库，处理2d版本的那些一来以外，还要pip install tensorflow-gpu==1.6
然后需要7.0版本的cudnn for cuda9（如果之前安装有其他版本的cudnn先删除相应的include/ 和 lib64/目录下的文件）  https://developer.nvidia.com/rdp/cudnn-archive
从以上网址下载cuDNN v7.0.5 Library for Linux，解压缩tar zxvf cudnn-9.0-linux-x64-v7.tgz
把include/ 和 lib64/目录下的文件拷贝到cuda的安装目录下的include/ 和 lib64里面
sudo cp -P include/cudnn.h /usr/local/cuda90/include
sudo cp -P libcudnn* /usr/local/cuda9/lib64
