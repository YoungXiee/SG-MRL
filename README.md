## Training
```python
# Train on Exchange-Rate
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model/exchange_rate-3.pt --data ./data/exchange_rate.txt --num_nodes 8 --batch_size 4 --epochs 50 --horizon 3
# Train on Traffic
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model/traffic-3.pt --data ./data/traffic.txt --num_nodes 862 --batch_size 4 --epochs 50 --horizon 3
# Train on Electricity
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model/electricity-3.pt --data ./data/electricity.txt --num_nodes 321 --batch_size 4 --epochs 50 --horizon 3
# Train on Nasdaq
CUDA_LAUNCH_BLOCKING=1 python train.py --save ./model/nasdaq-3.pt --data ./data/nasdaq.txt --num_nodes 82 --batch_size 4 --epochs 50 --horizon 3
```
