export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=17
export TORCH_COMPILE_DEBUG=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Create output directory
OUTPUT_DIR="path/to/test/images"
mkdir -p $OUTPUT_DIR

# Path configurations
MODEL_PATH="path/to/model"
TRAIN_DATA_PATH="path/to/train/data"
TEST_DATA_PATH="path/to/inference/conversation"
MASK_PATH="path/to/mask"
IMAGE_PATH="path/to/sample/test/image"

# Training parameters
NUM_EPOCHS=500
BATCH_SIZE=10
LEARNING_RATE=0.01
EVAL_INTERVAL=5
SEED=17

# Function to run multi-GPU training
torchrun --nproc_per_node=1 path/to/train/phi3 \
        --model_path $MODEL_PATH \
        --train_data_path $TRAIN_DATA_PATH \
        --test_data_path $TEST_DATA_PATH \
        --image_path $IMAGE_PATH \
        --mask_path $MASK_PATH \
        --output_dir $OUTPUT_DIR \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --eval_interval $EVAL_INTERVAL \
        --seed $SEED \
        --distributed \
        