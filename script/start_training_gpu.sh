SCRIPTPATH=`dirname $0`
cd $SCRIPTPATH/..
THEANO_FLAGS="device=gpu" python train.py