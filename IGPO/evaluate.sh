
set -e

# ============================================
# Environment Variables
# ============================================
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0
export PET_NODE_RANK=0

# ============================================
# User Configuration (modify these)
# ============================================
export project_name=${project_name:-"project_name"}
export experiment_name=${experiment_name:-"experiment_name"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
TEST_FILES=${TEST_FILES:-"./data/test.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results"}
EVAL_LOG_PATH=${EVAL_LOG_PATH:-"./eval_log"}
MAX_TURNS=${MAX_TURNS:-10}
SEARCH_ENGINE=${SEARCH_ENGINE:-"online_search"}  
N_GPUS=${N_GPUS:-8}

DATA_WRITING_PATH=${DATA_WRITING_PATH:-"./cache/task_queue/"}

# ============================================
# Create output directories
# ============================================
mkdir -p ${OUTPUT_DIR}
mkdir -p ${EVAL_LOG_PATH}

echo "============================================"
echo "IGPO Evaluation (val_only mode)"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Test data: ${TEST_FILES}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max turns: ${MAX_TURNS}"
echo "Search engine: ${SEARCH_ENGINE}"
echo "Data writing path: ${DATA_WRITING_PATH}"
echo "GPUs: ${N_GPUS}"
echo "============================================"

# ============================================
# Run Evaluation using main_ppo with val_only=true
# ============================================
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${TEST_FILES} \
    data.val_files=${TEST_FILES} \
    data.train_batch_size=32 \
    data.max_prompt_length=30767 \
    data.max_response_length=2000 \
    +data.max_model_len=32768 \
    +data.data_writing_path=${DATA_WRITING_PATH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=${MODEL_PATH} \
    critic.ppo_micro_batch_size_per_gpu=2 \
    algorithm.gamma=1.0 \
    +algorithm.info_gain_type=log_prob_diff \
    +algorithm.info_gain_norm_mode=separate \
    +algorithm.use_vectorized_gt_logprob=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=true \
    +trainer.val_only=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.validation_data_dir=${EVAL_LOG_PATH} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    agent_grpo.n=1 \
    max_turns=${MAX_TURNS} \
    search_engine=${SEARCH_ENGINE} \
    codeact_env_disabled=true \
    trainer.total_epochs=0 \
    2>&1 | tee ${OUTPUT_DIR}/eval.log

echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Metrics saved to: ${EVAL_LOG_PATH}"
echo "============================================"
