
Image 를 input으로 제공받는 모델을 서빙하고 테스트 해보겠습니다.  
OpenAI Vision API 와 호환이 가능하게 서빙을 하면, GPT-4V 나 GPT-4o로 구현된 LLM application 에 호환이 좋겠죠.


```bash
vllm serve microsoft/Phi-3.5-vision-instruct \
  --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=2
```

OpenAI Vision API 는 Chat Completion 스타일에 추가가 되어있죠, 그래서 vllm 으로 서빙시에 Chat Template 이 필수 입니다. 위에서 사용한 Phi-3.5-vision-instruct 모델은 chat template 이 내장이기 때문에 문제는 없지만, 다른 모델들은 확인을 해야합니다.  


아래와 같은 이미지를 입력하는 예시로 사용하면 chat completion openai API 로 바로 사용이 가능합니다.  

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Single-image input inference
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            # NOTE: The prompt formatting with the image token `<image>` is not needed
            # since the prompt will be processed automatically by the API server.
            {"type": "text", "text": "What’s in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }],
)
print("Chat completion output:", chat_response.choices[0].message.content)

# Multi-image input inference
image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

chat_response = client.chat.completions.create(
    model="microsoft/Phi-3.5-vision-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the animals in these images?"},
            {"type": "image_url", "image_url": {"url": image_url_duck}},
            {"type": "image_url", "image_url": {"url": image_url_lion}},
        ],
    }],
)
print("Chat completion output:", chat_response.choices[0].message.content)
```


> 참고!

https://github.com/vllm-project/vllm/pull/7916