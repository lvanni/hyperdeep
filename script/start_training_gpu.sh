SCRIPTPATH=`dirname $0`
cd $SCRIPTPATH/..
module load cuda/8.0
THEANO_FLAGS="device=gpu" python train.py