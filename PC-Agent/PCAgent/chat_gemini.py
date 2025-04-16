import time
from google import genai
from google.genai.types import GenerateContentConfig

# —— 配置 ——  
# 1) 用你的实际 API Key 或者在环境变量中配置 GOOGLE_API_KEY  
# 2) 如果你要走 Vertex AI，还可以加上 vertexai=True, project, location 等参数  
client = genai.Client(api_key="YOUR_API_KEY")  
MODEL = "gemini-2.0-flash-001"  


def inference_chat(chat_history, 
                   model: str = MODEL, 
                   temperature: float = 0.0, 
                   seed: int = 1234, 
                   max_output_tokens: int = 2048) -> str:
    """
    调用 Gemini API 生成回复（等价于原来的 inference_chat，temp=0.0, seed=1234）。
    
    chat_history: List[{"role": str, "parts": List[Part]}]
    """
    config = GenerateContentConfig(
        temperature=temperature,
        seed=seed,
        max_output_tokens=max_output_tokens
    )
    
    # 简单的重试逻辑
    while True:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=chat_history,
                config=config
            )
            return resp.text
        except Exception as e:
            print(f"Gemini API Error: {e}, retrying in 1s…")
            time.sleep(1)


def inference_chat_V2(chat_history, 
                      model: str = MODEL, 
                      temperature: float = 0.2, 
                      seed: int = 42, 
                      max_output_tokens: int = 2048) -> str:
    """
    调用 Gemini API 生成回复（等价于原来的 inference_chat_V2，temp=0.2, seed=42）。
    """
    config = GenerateContentConfig(
        temperature=temperature,
        seed=seed,
        max_output_tokens=max_output_tokens
    )
    
    while True:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=chat_history,
                config=config
            )
            return resp.text
        except Exception as e:
            print(f"Gemini API Error: {e}, retrying in 1s…")
            time.sleep(1)
