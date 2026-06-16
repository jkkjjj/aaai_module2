/home/kaijie/miniconda3/envs/emu3/bin/python ~/repo/iclr2/EvoTest/main.py \
  --game_name detective --rom_path jericho-games/ \
  --agent_type our \
  --llm_model gpt-oss-20b \
  --evolution_llm_model gpt-oss-20b \
  --eval_runs 10 \
  --enable_dacs \
  --dacs_debug



/home/kaijie/miniconda3/envs/emu3/bin/python ~/repo/iclr2/EvoTest/main.py \
  --game_name detective --rom_path jericho-games/ \
  --agent_type our \
  --llm_model gpt-oss-20b \
  --evolution_llm_model gpt-oss-20b \
  --eval_runs 10 \
  --enable_dacs \
  --dacs_top_n 8 \
  --dacs_select_k 4 \
  --dacs_alpha_relevance 1.0 \
  --dacs_beta_diversity 0.5 \
  --dacs_gamma_risk 0.7 \
  --dacs_debug





