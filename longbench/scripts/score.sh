### models
models=(
    "llama2-7b-hf" \
    "llama2-7b-hf-slimpajama-landmark" \
    "llama2-7b-hf-lminfinite" \
    "llama2-7b-hf-slimpajama-pi-32k" \
    "llama2-7b-hf-slimpajama-longlora-32k" \ 
    "llama2-7b-hf-ntk-frozen" \
    "llama2-7b-hf-slimpajama-ntk-32k" \
    "llama2-7b-hf-slimpajama-ntk-64k" \
    "llama-2-7b-hf-slimpajama-ntk-64k-2B" \
    "llama2-7b-hf-slimpajama-yarn-32k"
    )

### models test 4k
# models=(
#     "llama2-7b-hf-slimpajama-landmark-test4k" \
#     "llama2-7b-hf-lminfinite-test4k" \
#     "llama2-7b-hf-slimpajama-pi-32k-test4k" \
#     "llama2-7b-hf-slimpajama-longlora-32k-test4k"  \
#     "llama2-7b-hf-ntk-frozen-test4k" \
#     "llama2-7b-hf-slimpajama-ntk-64k-test4k" \
#     "llama-2-7b-hf-slimpajama-ntk-64k-2B-test4k" \
#     "llama2-7b-hf-slimpajama-yarn-32k-test4k"
# )

for MODEL_NAME in "${models[@]}"; 
do
python eval_trec.py --model ${MODEL_NAME}
done