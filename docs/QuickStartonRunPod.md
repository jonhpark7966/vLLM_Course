

## 직접 돌려보기!

A40 (48GB) x2 를 대여하겠습니다.  
대여는 runpod 에서 했고요, 서비스 이름에서 알수 있듯  `konuu/llm_ready:latest` docker image 를 올려 줍니다.  vllm image 를 안 올린 이유는, model을 처음부터 같이 띄워주는게 싫어서 입니다.

> (참고) RTX3090 (24GB) x 4 를 먼저 대여했는데 에러가 나서 안뜨는 군요... ;;

 개인적으로는 container 가 올라간 것이 아니라 하드웨어를 빌려주는 LambdaLabs 를 선호하는데, 요즘 GPU 가 동나서... 빌리기가 힘드네요.
 
![](./rsc/402.png)


서빙을 한번 해보죠, 환경변수 셋업을 한번 합니다.  meta-llama 는 허깅페이스에서 권한을 받아야만 사용이 가능하니, 미리 승인을 받아두시고 huggingface token을 등록합니다. 

``` bash
export HF_TOKEN=HF-TOKEN
```

이제 vLLM 서빙을 시작합니다. 

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```



Runpod ID 로 serving 이 되고 있는지 날려봅니다, 131k 입력을 받는 llama3.1 이 서빙 되고 있군요. 

```bash
❯ curl https://36g54goiy09px8-8000.proxy.runpod.net/v1/models

{"object":"list","data":[{"id":"meta-llama/Llama-3.1-8B-Instruct","object":"model","created":1729342516,"owned_by":"vllm","root":"meta-llama/Llama-3.1-8B-Instruct","parent":null,"max_model_len":131072,"permission":[{"id":"modelperm-f4239c4af18545faabe2190499d6b567","object":"model_permission","created":1729342516,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}%   
```


OpenAI compatible 한 서버니, chat completion api 를 호출합니다. 대답 토큰이 날라옵니다 !!

```bash
❯ curl https://36g54goiy09px8-8000.proxy.runpod.net/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'

{"id":"chat-076af67d9cff46659af30b617a44a490","object":"chat.completion","created":1729342638,"model":"meta-llama/Llama-3.1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I assist you today?","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":43,"total_tokens":53,"completion_tokens":10},"prompt_logprobs":null}% 
```


## Arguments

`vllm serve` 커맨드의 arguments 중, 알 필요가 있다고 생각 되는 부분들만 뽑아서 정리합니다.

`--model`
- 당연히 첫번째는 모델이고요, hugging face 주소를 입력하면 됩니다. huggingface 대신 modelscope 으로 갈 수도 있긴한데, 중국 쪽을 타겟으로 하지 않는 한 아마도 huggingface 면 충분하지 않을까 싶네요.


### 메모리 관련


`--max-model-len`
- 모델의 content 길이 입니다. 기본적으로는 모델에 설정된 config 를 따라갑니다. 그런데, 요즘 모델들은 과하게 긴 context length 를 가지고 있죠. (Llama-3.1-8B 는 131k 를 받을 수 있습니다. )
- Out-of-Memory 에러가 발생한 경우, 그리고 그렇게 큰 input을 받을 필요는 없는 경우, 그냥 context length를 제한 하면 메모리에 올려 서빙이 가능해집니다.


`--gpu-memory-utilization`
- GPU 메모리 사용량을 지정합니다. vLLM 의 핵심 알고리즘인 Paged Attention 특성상 메모리를 greedy 하게 가져가기 떄문에, 지정하면 그냥 다 먹습니다, 다른 GPU 메모리가 필요한 일이 있다면, 망가지겠죠.
- 이는 0 ~ 1 비율이고, 전체 메모리에 대한 비율이 아니라 남아있는 메모리에 대한 비율입니다. LLM 을 서빙하면 보통은 하나를 통쨰로 가져갈테니 default 값인 0.9 가 나쁘지 않은 것 같습니다.

### Multi GPU 관련

`--pipeline-parallel-size, -pp`
- 여러 개의 GPU 로 병렬 처리 합니다. parallelism level 은 layer 단위로요. high throughput 에 도움이 됩니다. 메모리가 부족한 경우에도 효과가 있습니다.

`--tensor-parallel-size, -tp`
- 여러 개의 GPU 로 병렬 처리 합니다. parallelism level 은 tensor 연산에 적용됩니다. 메모리가 부족한 경우에도 효과가 있습니다.

두 병렬와 단계에 대한 내용은 [Tensor parallelism vs Pipeline Parallelim](https://colossalai.org/docs/concepts/paradigms_of_parallelism/)  를 참조하시면 이해에 도움이 되실 것입니다.



### Quantization 관련


`--dtype`

- 데이터 타입을 설정합니다. auto, half, float16, bfloat16, float, float32 옵션 중 고르면 되는데, 요즘 오픈모델들은 대부분 bfloat16을 기반으로 하고 있고, 다르게 변경된 모델들이 있죠. 
- bfloat16은 주의할 점이 있는데, 옛날 nvidia gpu 들이 지원안합니다. Tesla v100 같은 GPU는 안하니까 호환성을 조심하세요.  

`--quantization, -q`
- Quantized 옵션입니다. 다음과 같은 후보들이 있죠. aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, modelopt, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, experts_int8, neuron_quant, ipex, None
- 참. 여기는 통일이 안되어서 갑갑한데요, 그래도 bnb 가 가장 많이 사용되는 것 같고요, 만들어진 모델을 잘 보고 맞춰주시면 됩니다. 직접 quantization 하실 분들은 이미 잘 아실테니 자세한 설명은 생략합니다.
- 

On RTX3090 Ubuntu

```
$ vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct

The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (41152)
```

```
$ vllm serve unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --quantization bitsandbytes --load-format bitsandbytes

The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (119712)
```
  


`--load-format`
- 모델의 파라미터들 포맷을 지정해줍니다, 꼭 Quantization 이랑만 관련이 있는 것은 아니지만, quantized 모델을 로드하는 경우에 위 -q 옵션이랑 잘 맞춰주지 않으면 에러가 잘 나기 떄문에 이 파트에 설명합니다.
- bnb 모델을 로드하시면 여기도 똑같이 bnb 맞춰주시면 됩니다.
- 가능 옵션들:  auto, pt, safetensors, npcache, dummy, tensorizer, sharded_state, gguf, bitsandbytes, mistral




## Metrics

vllm 을 서빙하면 `/metrics` endpoint 에서 지표들을 관찰 가능합니다.

```
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:num_requests_swapped Number of requests swapped to CPU.
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:cpu_cache_usage_perc CPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:cpu_cache_usage_perc gauge
vllm:cpu_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:cpu_prefix_cache_hit_rate CPU prefix cache block hit rate.
# TYPE vllm:cpu_prefix_cache_hit_rate gauge
vllm:cpu_prefix_cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} -1.0
# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache block hit rate.
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="meta-llama/Llama-3.1-8B-Instruct"} -1.0
# HELP vllm:avg_prompt_throughput_toks_per_s Average prefill throughput in tokens/s.
# TYPE vllm:avg_prompt_throughput_toks_per_s gauge
vllm:avg_prompt_throughput_toks_per_s{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:avg_generation_throughput_toks_per_s Average generation throughput in tokens/s.
# TYPE vllm:avg_generation_throughput_toks_per_s gauge
vllm:avg_generation_throughput_toks_per_s{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",cpu_offload_gb="0",enable_prefix_caching="False",gpu_memory_utilization="0.9",num_cpu_blocks="2048",num_gpu_blocks="2375",num_gpu_blocks_override="None",sliding_window="None",swap_space_bytes="4294967296"} 1.0
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0
```

참고로, vllm backend 를 사용하는 nvidia triton 컨테이너는 24.08 버전 부터 지원하니 참고하세요.
