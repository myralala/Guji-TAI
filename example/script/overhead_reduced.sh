mkdir -p /gemini/data-3/lrl/Know-MRI/results/experiments/2026-03-31/overhead_reduced_qwen3_8b

CUDA_VISIBLE_DEVICES=0,1 python /gemini/data-3/lrl/Know-MRI/example/script/guji_overhead_eval.py \
  --model_name Qwen/Qwen3-8B \
  --sample_limit 10 \
  --out_dir /gemini/data-3/lrl/Know-MRI/results/experiments/2026-03-31/overhead_reduced_qwen3_8b \
  --datasets 自动标点_chain_of_thought 缺字补全_chain_of_thought 传统文化问答_0_shot 关系抽取_0_shot \
  --methods Attribution "Attention Weights" FiNE "Logit Lens" \
  > /gemini/data-3/lrl/Know-MRI/results/experiments/2026-03-31/overhead_reduced_qwen3_8b/run.log 2>&1
