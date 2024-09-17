
methods=("llama2-7b-hf" "llama2-7b-hf-lminfinite" "llama2-7b-hf-ntk-frozen" "llama-2-7b-hf-slimpajama-pi-32k" \
    "llama-2-7b-hf-slimpajama-ntk-32k" "llama2-7b-hf-slimpajama-ntk-64k" "llama2-7b-hf-slimpajama-ntk-64k-2B" \
    "llama2-7b-hf-slimpajama-yarn-32k" "llama2-7b-hf-slimpajama-longlora-32k" "llama2-7b-hf-slimpajama-landmark")

for method in "${methods[@]}"; 
do
for length in 4096 8192 16384 32768 65536;
do
python eval/evaluate.py \
  --data_dir /results/${method}/synthetic/${length}/pred \
  --benchmark synthetic
done
done