$cmd = "cd /mnt/c/Users/kitti/Desktop/KrakenSK && source .venv/bin/activate && hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507"
wsl -e bash -lc "$cmd"
