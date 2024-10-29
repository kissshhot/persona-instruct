export OPENAI_API_KEY="sk-cBMDIQFHqCyqrGlSFb26F27dD9C4445d971a0673A73c7f9a"
export OPENAI_API_BASE="http://15.204.101.64:4000/v1"
export HF_ENDPOINT=https://hf-mirror.com
alpaca_eval --model_outputs '/home/dyf/data_generate/persona-instruct/evaluate/outputs/output.json' \
    --output_path '/home/dyf/data_generate/persona-instruct/evaluate/result/alpaca/'