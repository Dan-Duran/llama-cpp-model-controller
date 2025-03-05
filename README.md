##### Setup #####
nvidia-smi
mkdir /home/dan/llama
cd /home/dan/llama
wget https://github.com/ggml-org/llama.cpp/archive/refs/tags/b4823.zip
unzip b4823.zip
cd llama.cpp-b4823
(FAILED) cmake -B build -DGGML_CUDA=ON
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80"
cmake --build build --config Release -j 24
cd /home/dan/llama
mkdir models
cd models
wget https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q6_K_L.gguf

##### RUN ######
cd /home/dan/llama

CUDA_VISIBLE_DEVICES=0,1 /home/dan/llama/llama.cpp-b4823/build/bin/llama-server -m /home/dan/llama/models/DeepSeek-R1-Distill-Qwen-32B-Q6_K_L.gguf --threads 16 --port 8080 --host 0.0.0.0 -ngl 99  -c 32000 -sm row -np 3

To access the AI server's IP and port on any computer or mobile device:

http://192.168.xx.xxx:8080

