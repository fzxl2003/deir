# About
This repository implements __Discriminative-model-based Episodic Intrinsic Reward (DEIR)__, an exploration method for reinforcement learning that has been found to be particularly effective in environments with stochasticity and partial observability.
More details can be found in the original paper "_DEIR: Efficient and Robust Exploration through Discriminative-Model-Based Episodic Intrinsic Rewards_" ([arXiv preprint](https://arxiv.org/abs/2304.10770), to appear at [IJCAI 2023](https://ijcai-23.org/)).

Our PPO implementation is based on [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3/tree/master). In case you are mainly interested in the implementation of DEIR, its major components can be found at [src/algo/intrinsic_rewards/deir.py](https://github.com/swan-utokyo/deir/blob/main/src/algo/intrinsic_rewards/deir.py).

### Demos
Video demos of DEIR: (1) [Introduction & MiniGrid demo](https://youtu.be/VDnkFuGKZC0), (2) [ProcGen demo](https://youtu.be/ljmiXwKq1u8).

<img src="https://github.com/swan-utokyo/deir/assets/132561960/79235738-7a41-4a66-afb8-dc15325f40f9" width="480">

# Usage
### Installation
```commandline
conda create -n deir python=3.9
conda activate deir 
git clone https://github.com/swan-utokyo/deir.git
cd deir
python3 -m pip install -r requirements.txt
```

### Train DEIR on MiniGrid
Run the below command in the root directory of this repository to train a DEIR agent in the standard _DoorKey-8x8_ (MiniGrid) environment.
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=DoorKey-8x8
```

### Train DEIR on MiniGrid with advanced settings
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=DoorKey-8x8-ViewSize-3x3 \
  --can_see_walls=0 \
  --image_noise_scale=0.1
```

### Sample output in DoorKey-8x8
```
run:  0  iters: 1  frames: 8192  rew: 0.389687  rollout: 9.281 sec  train: 1.919 sec
run:  0  iters: 2  frames: 16384  rew: 0.024355  rollout: 9.426 sec  train: 1.803 sec
run:  0  iters: 3  frames: 24576  rew: 0.032737  rollout: 8.622 sec  train: 1.766 sec
run:  0  iters: 4  frames: 32768  rew: 0.036805  rollout: 8.309 sec  train: 1.776 sec
run:  0  iters: 5  frames: 40960  rew: 0.043546  rollout: 8.370 sec  train: 1.768 sec
run:  0  iters: 6  frames: 49152  rew: 0.068045  rollout: 8.337 sec  train: 1.772 sec
run:  0  iters: 7  frames: 57344  rew: 0.112299  rollout: 8.441 sec  train: 1.754 sec
run:  0  iters: 8  frames: 65536  rew: 0.188911  rollout: 8.328 sec  train: 1.732 sec
run:  0  iters: 9  frames: 73728  rew: 0.303772  rollout: 8.354 sec  train: 1.741 sec
run:  0  iters: 10  frames: 81920  rew: 0.519742  rollout: 8.239 sec  train: 1.749 sec
run:  0  iters: 11  frames: 90112  rew: 0.659334  rollout: 8.324 sec  train: 1.777 sec
run:  0  iters: 12  frames: 98304  rew: 0.784067  rollout: 8.869 sec  train: 1.833 sec
run:  0  iters: 13  frames: 106496  rew: 0.844819  rollout: 9.068 sec  train: 1.740 sec
run:  0  iters: 14  frames: 114688  rew: 0.892450  rollout: 8.077 sec  train: 1.745 sec
run:  0  iters: 15  frames: 122880  rew: 0.908270  rollout: 7.873 sec  train: 1.738 sec
```

### Train DEIR on ProcGen
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=procgen --game_name=ninja --total_steps=100_000_000 \
  --num_processes=64 --n_steps=256 --batch_size=2048 \
  --n_epochs=3 --model_n_epochs=3 \
  --learning_rate=1e-4 --model_learning_rate=1e-4 \
  --policy_cnn_type=2 --features_dim=256 --latents_dim=256 \
  --model_cnn_type=1 --model_features_dim=64 --model_latents_dim=256 \
  --policy_cnn_norm=LayerNorm --policy_mlp_norm=NoNorm \
  --model_cnn_norm=LayerNorm --model_mlp_norm=NoNorm \
  --adv_norm=0 --adv_eps=1e-5 --adv_momentum=0.9
```

### Example of Training Baselines
Please note that the default value of each option in `src/train.py` is optimized for DEIR. For now, when training other methods, please use the corresponding hyperparameter values specified in Table A1 (in our [arXiv preprint](https://arxiv.org/abs/2304.10770)). An example is `--int_rew_coef=3e-2` and `--rnd_err_norm=0` in the below command.
```commandline
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=NovelD \
  --env_source=minigrid \
  --game_name=DoorKey-8x8 \
  --int_rew_coef=3e-2 \
  --rnd_err_norm=0
```



######### 新参数训练 42 BlockedUnlockPickup Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=BlockedUnlockPickup \
  --run_id=42 \
  --save_model_path=./42BlockedUnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004402378806967045 \
  --batch_size=128 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.995 \
  --vf_coef=0.7225267431531684 \
  --ent_coef=6.451305148398007e-05 \
  --max_grad_norm=5 \
  --clip_range=0.1 \

######### 新参数训练 1024 BlockedUnlockPickup Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=BlockedUnlockPickup \
  --run_id=1024 \
  --save_model_path=./1024BlockedUnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004402378806967045 \
  --batch_size=128 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.995 \
  --vf_coef=0.7225267431531684 \
  --ent_coef=6.451305148398007e-05 \
  --max_grad_norm=5 \
  --clip_range=0.1 \

######### 新参数训练 5840 BlockedUnlockPickup Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=BlockedUnlockPickup \
  --run_id=5840 \
  --save_model_path=./5840BlockedUnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004402378806967045 \
  --batch_size=128 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.995 \
  --vf_coef=0.7225267431531684 \
  --ent_coef=6.451305148398007e-05 \
  --max_grad_norm=5 \
  --clip_range=0.1 \

######### 新参数训练 4649 BlockedUnlockPickup Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=BlockedUnlockPickup \
  --run_id=4649 \
  --save_model_path=./4649BlockedUnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004402378806967045 \
  --batch_size=128 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.995 \
  --vf_coef=0.7225267431531684 \
  --ent_coef=6.451305148398007e-05 \
  --max_grad_norm=5 \
  --clip_range=0.1 \


######### 新参数训练 314159 BlockedUnlockPickup Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=BlockedUnlockPickup \
  --run_id=314159 \
  --save_model_path=./314159BlockedUnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004402378806967045 \
  --batch_size=128 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.995 \
  --vf_coef=0.7225267431531684 \
  --ent_coef=6.451305148398007e-05 \
  --max_grad_norm=5 \
  --clip_range=0.1 \

######### 新参数训练 2986 BlockedUnlockPickup Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=BlockedUnlockPickup \
  --run_id=2986 \
  --save_model_path=./2986BlockedUnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004402378806967045 \
  --batch_size=128 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.995 \
  --vf_coef=0.7225267431531684 \
  --ent_coef=6.451305148398007e-05 \
  --max_grad_norm=5 \
  --clip_range=0.1 \


######### 新参数测试BlockedUnlockPickup xxx  tanh ########
PYTHONPATH=./ python3 src/test.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=1024 \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
--load_model_path=/root/project/deir/runs/1024BlockedUnlockPickupmodelold



######### 新参数训练 42 Unlock Tanh #########

PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=Unlock \
  --run_id=42 \
  --save_model_path=./42Unlockmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004993915310303568 \
  --batch_size=64 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.95 \
  --vf_coef=0.9234622679465284 \
  --ent_coef=5.368707796294674e-05 \
  --max_grad_norm=0.8 \
  --clip_range=0.3 \
######### 新参数训练 1024 Unlock Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=Unlock \
  --run_id=1024 \
  --save_model_path=./1024Unlockmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004993915310303568 \
  --batch_size=64 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.95 \
  --vf_coef=0.9234622679465284 \
  --ent_coef=5.368707796294674e-05 \
  --max_grad_norm=0.8 \
  --clip_range=0.3 \
######### 新参数训练 5840 Unlock Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=Unlock \
  --run_id=5840\
  --save_model_path=./5840Unlockmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004993915310303568 \
  --batch_size=64 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.95 \
  --vf_coef=0.9234622679465284 \
  --ent_coef=5.368707796294674e-05 \
  --max_grad_norm=0.8 \
  --clip_range=0.3 \

######### 新参数训练 4649 Unlock Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=Unlock \
  --run_id=4649\
  --save_model_path=./4649Unlockmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004993915310303568 \
  --batch_size=64 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.95 \
  --vf_coef=0.9234622679465284 \
  --ent_coef=5.368707796294674e-05 \
  --max_grad_norm=0.8 \
  --clip_range=0.3 \
######### 新参数训练 314159 Unlock Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=Unlock \
  --run_id=314159\
  --save_model_path=./314159Unlockmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004993915310303568 \
  --batch_size=64 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.95 \
  --vf_coef=0.9234622679465284 \
  --ent_coef=5.368707796294674e-05 \
  --max_grad_norm=0.8 \
  --clip_range=0.3 \

######### 新参数训练 2986 Unlock Tanh #########
PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=Unlock \
  --run_id=2986\
  --save_model_path=./2986Unlockmodelnew \
  --features_dim=128 \
  --learning_rate=0.0004993915310303568 \
  --batch_size=64 \
  --n_steps=2048 \
  --n_epochs=20 \
  --gamma=0.95 \
  --vf_coef=0.9234622679465284 \
  --ent_coef=5.368707796294674e-05 \
  --max_grad_norm=0.8 \
  --clip_range=0.3 \

######### 新参数测试Unlock xxx  tanh ########
PYTHONPATH=./ python3 src/test.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=1024 \
--features_dim=128 \
--learning_rate=0.0004993915310303568 \
--batch_size=64 \
--n_steps=2048 \
--n_epochs=20 \
--gamma=0.95 \
--vf_coef=0.9234622679465284 \
--ent_coef=5.368707796294674e-05 \
--max_grad_norm=0.8 \
--clip_range=0.3 \
--load_model_path=/root/project/deir/runs/1024BlockedUnlockPickupmodelold

########## 新参数训练UnlockPickup 42  relu ######## 

PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=UnlockPickup \
  --run_id=42\
  --save_model_path=./42UnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0011085443668731837 \
  --batch_size=256 \
  --n_steps=2048 \
  --n_epochs=10 \
  --gamma=0.9 \
  --vf_coef=0.423967217641348 \
  --ent_coef=2.5176363253597164e-08 \
  --max_grad_norm=0.6 \
  --clip_range=0.1 \

########## 新参数训练UnlockPickup 1024  relu ######## 

PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=UnlockPickup \
  --run_id=1024\
  --save_model_path=./1024UnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0011085443668731837 \
  --batch_size=256 \
  --n_steps=2048 \
  --n_epochs=10 \
  --gamma=0.9 \
  --vf_coef=0.423967217641348 \
  --ent_coef=2.5176363253597164e-08 \
  --max_grad_norm=0.6 \
  --clip_range=0.1 \
########## 新参数训练UnlockPickup 5840  relu ######## 

PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=UnlockPickup \
  --run_id=5840\
  --save_model_path=./5840UnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0011085443668731837 \
  --batch_size=256 \
  --n_steps=2048 \
  --n_epochs=10 \
  --gamma=0.9 \
  --vf_coef=0.423967217641348 \
  --ent_coef=2.5176363253597164e-08 \
  --max_grad_norm=0.6 \
  --clip_range=0.1 \
########## 新参数训练UnlockPickup 4649  relu ######## 

PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=UnlockPickup \
  --run_id=4649\
  --save_model_path=./4649UnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0011085443668731837 \
  --batch_size=256 \
  --n_steps=2048 \
  --n_epochs=10 \
  --gamma=0.9 \
  --vf_coef=0.423967217641348 \
  --ent_coef=2.5176363253597164e-08 \
  --max_grad_norm=0.6 \
  --clip_range=0.1 \

########## 新参数训练UnlockPickup 314159  relu ######## 

PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=UnlockPickup \
  --run_id=314159\
  --save_model_path=./314159UnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0011085443668731837 \
  --batch_size=256 \
  --n_steps=2048 \
  --n_epochs=10 \
  --gamma=0.9 \
  --vf_coef=0.423967217641348 \
  --ent_coef=2.5176363253597164e-08 \
  --max_grad_norm=0.6 \
  --clip_range=0.1 \

########## 新参数训练UnlockPickup 2986  relu ######## 

PYTHONPATH=./ python3 src/train.py \
  --int_rew_source=DEIR \
  --env_source=minigrid \
  --game_name=UnlockPickup \
  --run_id=2986\
  --save_model_path=./2986UnlockPickupmodelnew \
  --features_dim=128 \
  --learning_rate=0.0011085443668731837 \
  --batch_size=256 \
  --n_steps=2048 \
  --n_epochs=10 \
  --gamma=0.9 \
  --vf_coef=0.423967217641348 \
  --ent_coef=2.5176363253597164e-08 \
  --max_grad_norm=0.6 \
  --clip_range=0.1 \

########## 新参数测试UnlockPickup xxxx  relu ########
PYTHONPATH=./ python3 src/test.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=1024 \
--features_dim=128 \
--learning_rate=0.0011085443668731837 \
--batch_size=256 \
--n_steps=2048 \
--n_epochs=10 \
--gamma=0.9 \
--vf_coef=0.423967217641348 \
--ent_coef=2.5176363253597164e-08 \
--max_grad_norm=0.6 \
--clip_range=0.1 \
--load_model_path=/root/project/deir/runs/1024BlockedUnlockPickupmodelold




#################### 新参数训练 RedBlueDoor tanh 42 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=42 \
--save_model_path=./42RedBlueDoornew \
--features_dim=128 \
--learning_rate=0.0006600625492645438 \
--batch_size=256 \
--n_steps=2048 \
--n_epochs=10 \
--gamma=0.9 \
--vf_coef=0.8991719073875422 \
--ent_coef=5.3841020764888015e-08 \
--max_grad_norm=1 \
--clip_range=0.1 \

#################### 新参数训练 RedBlueDoor tanh 1024 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=1024 \
--save_model_path=./1024RedBlueDoornew \
--features_dim=128 \
--learning_rate=0.0006600625492645438 \
--batch_size=256 \
--n_steps=2048 \
--n_epochs=10 \
--gamma=0.9 \
--vf_coef=0.8991719073875422 \
--ent_coef=5.3841020764888015e-08 \
--max_grad_norm=1 \
--clip_range=0.1 \

#################### 新参数训练 RedBlueDoor tanh 5840 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=5840 \
--save_model_path=./5840RedBlueDoornew \
--features_dim=128 \
--learning_rate=0.0006600625492645438 \
--batch_size=256 \
--n_steps=2048 \
--n_epochs=10 \
--gamma=0.9 \
--vf_coef=0.8991719073875422 \
--ent_coef=5.3841020764888015e-08 \
--max_grad_norm=1 \
--clip_range=0.1 \

#################### 新参数训练 RedBlueDoor tanh 4649 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=4649 \
--save_model_path=./4649RedBlueDoornew \
--features_dim=128 \
--learning_rate=0.0006600625492645438 \
--batch_size=256 \
--n_steps=2048 \
--n_epochs=10 \
--gamma=0.9 \
--vf_coef=0.8991719073875422 \
--ent_coef=5.3841020764888015e-08 \
--max_grad_norm=1 \
--clip_range=0.1 \

#################### 新参数训练 RedBlueDoor tanh 314159 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=314159 \
--save_model_path=./314159RedBlueDoornew \
--features_dim=128 \
--learning_rate=0.0006600625492645438 \
--batch_size=256 \
--n_steps=2048 \
--n_epochs=10 \
--gamma=0.9 \
--vf_coef=0.8991719073875422 \
--ent_coef=5.3841020764888015e-08 \
--max_grad_norm=1 \
--clip_range=0.1 \

#################### 新参数训练 RedBlueDoor tanh 2986 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=2986 \
--save_model_path=./2986RedBlueDoornew \
--features_dim=128 \
--learning_rate=0.0006600625492645438 \
--batch_size=256 \
--n_steps=2048 \
--n_epochs=10 \
--gamma=0.9 \
--vf_coef=0.8991719073875422 \
--ent_coef=5.3841020764888015e-08 \
--max_grad_norm=1 \
--clip_range=0.1 \

## 还需要改激活函数tanh  /root/project/deir/src/algo/ppo_model.py
## 需要手动调整tensorboard的路径 /root/project/deir/src/algo/ppo_rollout.py







## 旧参数训练

#################### 旧参数训练 BlockedUnlockPickup Relu 1024 ####################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=1024 \
--save_model_path=./1024BlockedUnlockPickupmodelold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \


#################### 旧参数训练BlockUnlockPickup relu 314519####################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=314159 \
--save_model_path=./314159BlockedUnlockPickupmodelold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 BlockedUnlockPickup Relu 4649 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=4649 \
--save_model_path=./4649BlockedUnlockPickupmodelold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 BlockedUnlockPickup Relu 5840 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=5840 \
--save_model_path=./5840BlockedUnlockPickupmodelold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 BlockedUnlockPickup Relu 42 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=42 \
--save_model_path=./42BlockedUnlockPickupmodelold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 BlockedUnlockPickup Relu 2986 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=2986 \
--save_model_path=./2986BlockedUnlockPickupmodelold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数测试 BlockedUnlockPickup Relu ##################
PYTHONPATH=./ python3 src/test.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=BlockedUnlockPickup \
--run_id=1024 \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
--load_model_path=/root/project/deir/runs/1024BlockedUnlockPickupmodelold



## 还需要改激活函数relu  /root/project/deir/src/algo/ppo_model.py
## 需要手动调整tensorboard的路径 /root/project/deir/src/algo/ppo_rollout.py

#################### 旧参数训练 Unlock Relu 42 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=42 \
--save_model_path=./42Unlockold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 Unlock Relu 1024 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=1024 \
--save_model_path=./1024Unlockold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 Unlock Relu 5840 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=5840 \
--save_model_path=./5840Unlockold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 Unlock Relu 4649 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=4649 \
--save_model_path=./4649Unlockold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 Unlock Relu 314159 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=314159 \
--save_model_path=./314159Unlockold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 Unlock Relu 2986 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=2986 \
--save_model_path=./2986Unlockold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

########## 旧参数测试Unlock Relu ##################
PYTHONPATH=./ python3 src/test.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=Unlock \
--run_id=1024 \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
--load_model_path=/root/project/deir/runs/1024BlockedUnlockPickupmodelold

#################### 旧参数训练 UnlockPickup Relu 42 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=42 \
--save_model_path=./42UnlockPickupold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 UnlockPickup Relu 1024 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=1024 \
--save_model_path=./1024UnlockPickupold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 UnlockPickup Relu 5840 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=5840 \
--save_model_path=./5840UnlockPickupold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 UnlockPickup Relu 4649 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=4649 \
--save_model_path=./4649UnlockPickupold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 UnlockPickup Relu 314159 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=314159 \
--save_model_path=./314159UnlockPickupold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 UnlockPickup Relu 2986 ################## 
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=2986 \
--save_model_path=./2986UnlockPickupold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

######### 旧参数测试 UnlockPickup Relu 
PYTHONPATH=./ python3 src/test.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=UnlockPickup \
--run_id=1024 \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
--load_model_path=/root/project/deir/runs/1024BlockedUnlockPickupmodelold




#################### 旧参数训练 RedBlueDoor relu 42 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=42 \
--save_model_path=./42RedBlueDoorold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \
#################### 旧参数训练 RedBlueDoor relu 1024 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=1024 \
--save_model_path=./1024RedBlueDoorold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 RedBlueDoor relu 5840 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=5840 \
--save_model_path=./5840RedBlueDoorold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 RedBlueDoor relu 4649 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=4649 \
--save_model_path=./4649RedBlueDoorold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 RedBlueDoor relu 314159 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=314159 \
--save_model_path=./314159RedBlueDoorold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \

#################### 旧参数训练 RedBlueDoor relu 2986 ##################
PYTHONPATH=./ python3 src/train.py   --int_rew_source=DEIR   --env_source=minigrid   --game_name=MiniGrid-RedBlueDoors-8x8-v0 \
--run_id=2986 \
--save_model_path=./2986RedBlueDoorold \
--features_dim=64 \
--learning_rate=3e-4 \
--batch_size=512 \
--n_steps=512 \
--n_epochs=4 \
--gamma=0.99 \
--vf_coef=0.5 \
--ent_coef=0.01 \
--max_grad_norm=0.5 \
--clip_range=0.2 \


Unlock 新参数
42：0.96
1024：0.91
5840：1
4649：0.98
314159：0.91
2986

Unlock 旧参数
均为1


BlockedUnlockPickup 新参数
42:0
1024:0
5840:0
4649:0
314159:0
2986

BlockedUnlockPickup 旧参数
42:0
1024:1
5840:0
4649:0
314159:1
2986


UnlockPickup 新参数
42:0.89
1024:0.91
5840:0.86
4649:0.979
314159:0.99
2986

UnlockPickup 旧参数
42:1
1024:1
5840:1
4649:1
314159:1
2986

RedBlueDoor 新参数
42:
1024:
5840:
4649:
314159:
2986:

RedBlueDoor 旧参数
42:
1024:
5840:
4649:
314159:
2986: