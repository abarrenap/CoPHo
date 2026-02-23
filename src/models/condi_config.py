# config.py
import torch

# 超参数配置
LEARNING_RATE = 1.5e-4      # 学习率
NUM_EPOCHS = 50           # 训练 epoch 数
BATCH_SIZE = 256            # 批训练大小
HIDDEN_DIM = 256            # 图神经网络隐藏特征维度
NUM_LAYERS = 6             # 图神经网络层数
DROPOUT = 0.1              # Dropout 比例（用于 GAT 层）
EVAL_BATCH_SIZE = 10

# conditional settings
enable_condition = True
condi_delta = 1
condition_weight = 1
sampling_multiplier = 2

# GIN embedding settings
gin_use_pretrained = True
gin_model_path = "/Users/aimarbarrenapol/Documents/EHU/TFG/CoPHo/src/weights/random_gin.pt"

homo_condition_phase_s = 0.6
homo_condition_phase_e = 1.0

condition_phase_s = homo_condition_phase_s
condition_phase_e = homo_condition_phase_e
condition_mode = 'top+'
condition_is_undirected = True
condition_target = ["embedding"]
condition_threshold = 0.1

all_condition_phase_s = 1.2
all_condition_phase_e = 1.0