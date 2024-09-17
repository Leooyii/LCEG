
methods=("llama2-7b-hf")

for method in "${methods[@]}"; 
do
for length in 4096 8192 16384 32768 65536;
do
python eval/evaluate.py \
  --data_dir /results/${method}/synthetic/${length}/pred \
  --benchmark synthetic
done
done