# ntk 32k
# MODEL_NAME=llama-2-7b-hf-slimpajama-ntk-32k
# MODEL_PATH=

# ntk 64k
# MODEL_NAME=llama-2-7b-hf-slimpajama-ntk-64k
# MODEL_PATH=

# ntk frozen
# MODEL_NAME=llama-2-7b-hf-ntk-frozen
# MODEL_PATH=

# ntk-64k-2B
MODEL_NAME=llama-2-7b-hf-slimpajama-ntk-64k-2B
MODEL_PATH=

CONTEXT_SIZE=4096


for SEQ_LEN in 2048 4096 8192 16384 32768 65536;
do
    python eval_ntk.py \
        --seq_len $SEQ_LEN \
        --context_size $CONTEXT_SIZE \
        --batch_size 1 \
        --base_model $MODEL_PATH \
        --model_name $MODEL_NAME \
        --data_path data/pg19/test.bin \
        --output_dir results/pg19/${MODEL_NAME}_${SEQ_LEN}_pg19.json

    python eval_ntk.py \
        --seq_len $SEQ_LEN \
        --context_size $CONTEXT_SIZE \
        --batch_size 1 \
        --base_model $MODEL_PATH \
        --model_name $MODEL_NAME \
        --data_path data/proof_file/test_sampled_data.bin \
        --output_dir results/proof_file/${MODEL_NAME}_${SEQ_LEN}_proof_file.json
done
