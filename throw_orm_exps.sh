
PRMS=(
    greedy
    Skywork/Skywork-Reward-Llama-3.1-8B-v0.2
    # Skywork/Skywork-Reward-Gemma-2-27B-v0.2
    Qwen/Qwen2.5-Math-PRM-7B
)


LLMs=(
    meta-llama/Llama-3.1-8B-Instruct
    # Qwen/Qwen2.5-7B-Instruct
    microsoft/Phi-4-mini-Instruct
    # microsoft/phi-4
)

BUDGET=(8 16 32 64 128)
mkdir -p orm_full_logs

for budget in ${BUDGET[@]}; do
    for PRM in ${PRMS[@]}; do
        for MODEL in ${LLMs[@]}; do
            MODEL_NAME=$(echo ${MODEL} | sed 's/.*\///')
            PRM_NAME=$(echo ${PRM} | sed 's/.*\///')
            if [ "${PRM_NAME}" == "greedy" ]; then  
                LOG_FILE="orm_full_logs/model_${MODEL_NAME}_prm_${PRM_NAME}.log"
            else
                LOG_FILE="orm_full_logs/model_${MODEL_NAME}_prm_${PRM_NAME}_budget_${budget}.log"
            fi
            # Skip if the log file already exists
            if [ -f "${LOG_FILE}" ]; then
                echo "Skipping ${MODEL_NAME} with ${PRM_NAME} at budget ${budget} - log file already exists"
                continue
            fi
            
            echo "Running ${MODEL_NAME} with ${PRM_NAME} at budget ${budget}"
            if [ "${PRM_NAME}" == "greedy" ]; then  
                CUDA_VISIBLE_DEVICES=6,7 python main.py --model-name ${MODEL} --sampling-method greedy > ${LOG_FILE} 2>&1
            else
                CUDA_VISIBLE_DEVICES=6,7 python main.py --model-name ${MODEL} --prm-path ${PRM} --sampling-method best-of-n --test-time-compute-budget ${budget} > ${LOG_FILE} 2>&1
            fi
        done
    done
done

