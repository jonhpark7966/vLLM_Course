# LoRA Adapters

vLLM base model에 [LoRA adapters](https://arxiv.org/abs/2106.09685) 를 올리겠습니다.  
vLLM model 들이 `SupportsLoRA` 룰 구현해 뒀으면, LoRA를 붙여 쓸 수 있습니다.  

로라는 request 마다 적용 시켜서 서빙할 때 좋아요.

SQL 용 LoRA adapter 를 다운 받아보겠습니다.

```python
from huggingface_hub import snapshot_download

sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")
```

```bash
huggingface-cli download yard1/llama-2-7b-sql-lora-test
```

부모 모델인 Llama-2-7b-hf 모델과 함께 서빙을 해봅니다.

> commit ID는 바뀔 수 있으니 꼭 확인하세요!

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-lora \
    --lora-modules sql-lora=$HOME/.cache/huggingface/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/
```


completion endpoint로 요청을 날릴 때, model에 lora adapter 이름을 넣어주면 됩니다.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```


> 24년 10월 기준, 필자의 개인적인 의견.
>  좋은 lora 모델을 찾기도 힘들고, 효과를 본 경험이 없습니다. diffusion 계열에서는 많이 좋은 모습을 보았는데, llm 에서는 쉽지 않네요. 특정 목적으로 직접 lora를 만들어봐야 더 판단이 가능할 듯 합니다.