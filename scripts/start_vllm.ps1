$cmd = "cd /mnt/c/Users/kitti/Desktop/KrakenSK && source .venv/bin/activate && python -m vllm.entrypoints.openai.api_server --model models/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 8000 --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.9"
wsl -e bash -lc "$cmd"
