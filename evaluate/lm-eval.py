git clone git@github.com:huggingface/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout adding_all_changess
pip install -e .[math,ifeval,sentencepiece]
lm-eval --model_args="pretrained=<your_model>,revision=<your_model_revision>,dtype=<model_dtype>" --tasks=leaderboard  --batch_size=auto --output_path=<output_path>
--apply_chat_templatefewshot_as_multiturn
https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about