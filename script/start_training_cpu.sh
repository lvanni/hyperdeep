SCRIPTPATH=`dirname $0`
cd $SCRIPTPATH/..
THEANO_FLAGS="device=cpu" python train.py