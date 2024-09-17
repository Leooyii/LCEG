### landmark
MODEL_NAME=llama-2-7b-hf-slimpajama-landmark
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 

### lminfinite
MODEL_NAME=llama-2-7b-hf-lminfinite
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 


### longlora-32k
MODEL_NAME=llama-2-7b-hf-slimpajama-longlora-32k
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 

### ntk-frozen
MODEL_NAME=llama-2-7b-hf-ntk-frozen
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 

### ntk-32k
MODEL_NAME=llama-2-7b-hf-slimpajama-ntk-32k
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 

### ntk-64k
MODEL_NAME=llama-2-7b-hf-slimpajama-ntk-64k
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 

### yarn-32k
MODEL_NAME=llama-2-7b-hf-slimpajama-yarn-32k
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 

### pi-32k
MODEL_NAME="llama-2-7b-hf-slimpajama-pi-32k"
MODEL_PATH=
python -u needle_in_haystack.py --s_len 0 --e_len 65536\
    --model_provider LLaMA \
    --model_path ${MODEL_PATH} \
    --test_name ${MODEL_NAME} 

