CONFIG=$1
GPUS=$2
LOG_DIR=$3
MODEL_NAME=$(basename "$(dirname $CONFIG)")

PORT=10902
while ! nc -z localhost $PORT; do
  PORT=$((PORT + 1))
done

while : # auto-resume: the code sometimes crash due to bug of gloo on some gpus
do
torchrun --nproc_per_node=$GPUS \
        --master_port=$PORT \
    train.py --c $CONFIG --model $MODEL_NAME --log_dir $LOG_DIR | tee $LOG_DIR/train.log | while read line; do echo "$line"; done

for PID in $(ps -aux | grep $CONFIG | grep python | awk '{print $2}')
do
    echo $PID
    kill -9 $PID
done
sleep 30
done
