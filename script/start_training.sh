#oarsub -p "gpu='YES'" -l /gpunum=1 /home/lvanni/start_training_gpu.sh

SCRIPTPATH=`dirname $0`
cd $SCRIPTPATH/hyperdeep

source /etc/profile.d/modules.sh
module load cuda/8.0
module load cudnn/5.1-cuda-8.0

time THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=0.8 python train.py
