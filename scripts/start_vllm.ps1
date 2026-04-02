$cmd = @'
cd /mnt/c/Users/kitti/Desktop/KrakenSK
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6
'@

wsl -e bash -lc $cmd
