
1.1: python analyze_expert_data.py

1.2: 

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant_proper --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --learning_rate 4e-3 --n_layers 5 --video_log_freq -1 --eval_batch_size 5000

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name bc_Humanoid --n_iter 1 --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --learning_rate 4e-3 --n_layers 5 --video_log_freq -1 --eval_batch_size 5000


1.3: python simulate_learning_rate_experiment.py


2.1:

python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 8 --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --n_layers 3 --video_log_freq -1


python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/HalfCheetah.pkl  --env_name HalfCheetah-v2 --exp_name dagger_halfcheetah --n_iter 8 --do_dagger --expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl --n_layers 3 --video_log_freq -1

python -m tensorboard.main --logdir data

(then get experiment full name and change values in create_dagger_figure.py
e.g.     
ant_exp = "q2_dagger_ant_final1_Ant-v2_18-09-2025_23-41-07"
halfcheetah_exp = "q2_dagger_halfcheetah_final1_HalfCheetah-v2_18-09-2025_23-42-28"
)

python create_dagger_figure.py