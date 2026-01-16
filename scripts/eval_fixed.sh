cd "$(dirname "$0")/.."
root_dir=.
echo $PWD

# ============ API配置 ============
# 方式1: 使用硅基流动API (推荐)
export SILICONFLOW_VL_API_KEY="sk-bjlnvdonkrqbntdxundoaxdjfyqmynjxwzmmvhwztwngguln"  # 视觉模型API密钥
export SILICONFLOW_TEXT_API_KEY="sk-guuejboscbyteqiehtivheyrylijxzwdfdocorcofruomgzt"  # 文本模型API密钥
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
export SILICONFLOW_VL_MODEL="Qwen/Qwen3-VL-32B-Instruct"  # 或 "zai-org/GLM-4.6V"
export SILICONFLOW_TEXT_MODEL="deepseek-ai/DeepSeek-V3.2"  # 或 "Qwen/Qwen2.5-72B-Instruct"

# 方式2: 使用通义千问API (备用)
# export QWEN_API_KEY="your-qwen-api-key"
# export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# 注意：如果设置了SILICONFLOW_*环境变量，将优先使用硅基流动API

CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/eval_2.py \
    --maxActions 150 \
    --eval_save_path $root_dir/logs/scene \
    --dataset_path /home/xxl/下载/WJA/DATASET/valset/Slum.json \
    --is_fixed  true\
    --gpu_id 0 \
    --batchSize 4 \
    --simulator_tool_port 30000
