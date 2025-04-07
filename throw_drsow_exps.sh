
PRMS=(
    drsow
    # unnormalized_drsow
)


LLMs=(
    meta-llama/Llama-3.1-8B-Instruct
    # Qwen/Qwen2.5-7B-Instruct
    microsoft/Phi-4-mini-Instruct
    # microsoft/phi-4
)

# BUDGET=(8 16 32 64 128)
BUDGET=(4 8 16 32 64)
mkdir -p 32bit_drsow_results_full_logs
for budget in ${BUDGET[@]}; do  
    for MODEL in ${LLMs[@]}; do
        for PRM in ${PRMS[@]}; do
            MODEL_NAME=$(echo ${MODEL} | sed 's/.*\///')
            PRM_NAME=$(echo ${PRM} | sed 's/.*\///')
            LOG_FILE="32bit_drsow_results_full_logs/model_${MODEL_NAME}_prm_${PRM_NAME}_budget_${budget}.log"

            # Skip if the log file already exists
            if [ -f "${LOG_FILE}" ]; then
                echo "Skipping ${MODEL_NAME} with ${PRM_NAME} at budget ${budget} - log file already exists"
                continue
            fi
            
            echo "Running ${MODEL_NAME} with ${PRM_NAME} at budget ${budget}"

            CUDA_VISIBLE_DEVICES=0 python main.py --model-name ${MODEL} --prm-path ${PRM} --sampling-method best-of-n --test-time-compute-budget ${budget} > ${LOG_FILE} 2>&1
        done
    done
done

