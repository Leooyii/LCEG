# selfextend
# MODEL_NAME=llama-2-7b-hf-selfextend
# MODEL_PATH=/nas/shared/NLP_A100/luyi/longdojo/models/llama2-7b-hf

MODEL_NAME=phi-2-selfextend
MODEL_PATH=/nas/shared/NLP_A100/luyi/longdojo/models/phi-2


CONTEXT_SIZE=32768
CPFS_PATH=/nas/shared/NLP_A100/luyi

for SEQ_LEN in 2048 4096 8192 16384 32768 65536;
do
# JOB_NAME_1=luyi-longdojo-ppl-pg19-${MODEL_NAME}-${SEQ_LEN}
# COMMAND_1="/nas/shared/NLP_A100/luyi/anaconda3/envs/longdojo/bin/python /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/eval_selfextend.py \
#         --seq_len $SEQ_LEN \
#         --context_size $CONTEXT_SIZE \
#         --batch_size 1 \
#         --base_model $MODEL_PATH \
#         --data_path /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/data/pg19/test.bin \
#         --output_dir /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/results/pg19/${MODEL_NAME}_${SEQ_LEN}_pg19_new.json \
#          > /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/logs/pg19/${MODEL_NAME}_${SEQ_LEN}_pg19_new.log 2>&1 ;"


# ${CPFS_PATH}/dlc submit pytorchjob \
#     --command "$COMMAND_1" \
#     --name "${JOB_NAME_1}" \
#     --priority "2" \
#     --workers "1" \
#     --worker_cpu "10" \
#     --worker_gpu "1" \
#     --worker_image "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/easyanimate:1.1.1-pytorch2.2.0-gpu-py310-cu118-ubuntu22.04" \
#     --worker_memory "128Gi" \
#     --worker_shared_memory "0Gi" \
#     --workspace_id "28245" \
#     --resource_id "quotau1syyub2gtf" \
#     --data_sources "d-8ue5omsmi6skek4zxn,d-70gv19u17quzbvqv3x,d-liurq4v8ke15rnp6pp,d-yiekt7ms8m7yqa3a1e,d-o0xjmpubpkbx3wl4ci"
    
JOB_NAME_2=luyi-longdojo-ppl-proof_file-${MODEL_NAME}-${SEQ_LEN}
COMMAND_2="/nas/shared/NLP_A100/luyi/anaconda3/envs/longdojo/bin/python /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/eval_selfextend.py \
        --seq_len $SEQ_LEN \
        --context_size $CONTEXT_SIZE \
        --batch_size 1 \
        --base_model $MODEL_PATH \
        --data_path /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/data/proof_file/test_sampled_data.bin \
        --output_dir /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/results/proof_file/${MODEL_NAME}_${SEQ_LEN}_proof_file.json \
         > /nas/shared/NLP_A100/luyi/longdojo/eval_perplexity/logs/proof_file/${MODEL_NAME}_${SEQ_LEN}_proof_file.log 2>&1 "
${CPFS_PATH}/dlc submit pytorchjob \
    --command "$COMMAND_2" \
    --name "${JOB_NAME_2}" \
    --priority "2" \
    --workers "1" \
    --worker_cpu "10" \
    --worker_gpu "1" \
    --worker_image "dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/easyanimate:1.1.1-pytorch2.2.0-gpu-py310-cu118-ubuntu22.04" \
    --worker_memory "128Gi" \
    --worker_shared_memory "0Gi" \
    --workspace_id "28245" \
    --resource_id "quotau1syyub2gtf" \
    --data_sources "d-8ue5omsmi6skek4zxn,d-70gv19u17quzbvqv3x,d-liurq4v8ke15rnp6pp,d-yiekt7ms8m7yqa3a1e,d-o0xjmpubpkbx3wl4ci"

done