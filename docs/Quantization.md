
24년 10월 기준, 가장 대중적으로 사용 되는 Quantization 기법은 BitsAndBytes 로 보입니다.  
AWQ, GPTQ 등 다양한 quantization 기법과 포맷들이 나오고 있어요.  


## BitsAndBytes

먼저 bitsandbytes 를 설치해줍니다. 

```bash
pip install bitsandbytes>=0.44.0
```


그리고, bitsandbytes 로 만들어진 quantized model을 다운받아 서빙하면 끝입니다.

```bash
$ vllm serve unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --quantization bitsandbytes --load-format bitsandbytes
```

unsloth 에서 만들어준 4bit 모델을 올려봤습니다. Unsloth 에서 유명한 모델들은 다 이렇게 quantize 해서 올려주니 사용하기가 아주 편합니다.  
그리고, 직접 quantization을 해서 모델을 자르고 싶으실 떄에도 unsloth를 추천합니다. 개인적으로 사용경험이 좋습니다. 쉬워서요.  


> 참고. 메모리 사용량 비교

제 개인 3090 머신에서 테스트한 결과입니다.
8B 모델은 기본 상태에서 `KV cache (41152)` 밖에 메모리에 모델에 못 올려서 context 줄이거나 해야합니다.

```
The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (41152)
```


unsloth 에서 만들어준 4bit quantized model 의 경우 역시 메모리가 모잘라서 올리지 못하지만, `KV cache (119712)` 에러 메세지에 보이듯 더 많이 올릴 수 있습니다. 더 크게 context length를 사용할 수 있죠.

```
$ vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct
```

```
The model's max seq len (131072) is larger than the maximum number of tokens that can be stored in KV cache (119712)```
```


## AutoAWQ

bnb 랑 방법은 똑같습니다. AWQ 설치해주고, AWQ로 잘려진 모델을 서빙하면 됩니다.

```
pip install autoawq
```

```bash
$ vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --quantization awq
```


A40 (48GB) 기준 올라가지 않던 32B 모델이 AWQ 와 함께라면 구동이 가능합니다 !!



## FP8 와 BF16

최근 몇년새에 NVIDIA 가 지원하기 시작한 quantization 기법들 입니다. 하드웨어 지원을 통해 정직한 (?) 성능향상을 이뤄냅니다.