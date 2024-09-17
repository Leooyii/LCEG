MODEL_NAME=llama-2-7b-hf-landmark
MODEL_PATH=


for SEQ_LEN in 2048 4096 8192 16384 32768 65536;
do
python eval_landmark.py \
        --seq_len $SEQ_LEN \
        --batch_size 1 \
        --base_model $MODEL_PATH \
        --data_path data/pg19/test.bin \
        --output_dir results/pg19/${MODEL_NAME}_${SEQ_LEN}_pg19.json 


python eval_landmark.py \
        --seq_len $SEQ_LEN \
        --batch_size 1 \
        --base_model $MODEL_PATH \
        --data_path data/proof_file/test_sampled_data.bin \
        --output_dir results/proof_file/${MODEL_NAME}_${SEQ_LEN}_proof_file.json 
done