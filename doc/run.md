

我想开几个tmux 游戏名有Detective Library Zork1 Zork3 Balances Temple Deephome Ztuu Ludicorp Pentari （全小写）

export OPENAI_API_KEY="9975d0f6-0228-4fae-a36a-1a484aae01bf"
export OPENAI_BASE_URL="https://sd80rokkc19oii7djp10g.apigateway-cn-beijing.volceapi.com/v1/"
export OPENAI_API_BASE="$OPENAI_BASE_URL"

/home/kaijie/miniconda3/envs/emu3/bin/python ~/repo/aaai1/module2/EvoTest/main.py \
  --game_name detective \
  --rom_path jericho-games/ \
  --agent_type our \
  --llm_model gpt-oss-20b \
  --evolution_llm_model gpt-oss-20b \
  --eval_runs 10 \
  --enable_dacs \
  --dacs_debug













