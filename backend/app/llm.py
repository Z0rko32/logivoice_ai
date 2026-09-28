import sys
import time
from openai import AsyncOpenAI
import os

from app.rag import rag_db

OLLAMA_API_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

client = AsyncOpenAI(
    base_url=OLLAMA_API_URL,
    api_key='ollama',
)

SYSTEM_PROMPT = """Ты — "LogiVoice", автоматизированный диспетчер логистической компании. Твоя задача — принимать экстренные сообщения водителей и выдавать четкие инструкции.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Игнорируй эмоциональный тон водителя. Отвечай кратко, строго по регламенту радиообмена ("Принял", конкретное указание).
2. ЗАПРЕЩЕНО давать общие советы, задавать встречные вопросы или советовать бытовые вещи. 
3. Твоя главная цель — выдать точный факт из БАЗЫ ЗНАНИЙ (номера телефонов, адреса TIR-паркингов, суммы лимитов, имена ответственных).
4. Если в БАЗЕ ЗНАНИЙ нет ответа, ответь: "Принял. В базе регламент отсутствует, передаю информацию старшему логисту. Ожидай."
"""

async def generate_response(user_text: str) -> tuple[str, str]:
    if not user_text:
        return "Я не услышал текст, повтори, пожалуйста.", ""

    print(f"\n[LLM] Начинаю генерацию ответа через Llama 3.1:8B...")
    start_time = time.time()

    context = rag_db.search_context(user_text)
    
    DYNAMIC_PROMPT = SYSTEM_PROMPT
    if context:
        DYNAMIC_PROMPT += f"\n\n[БАЗА ЗНАНИЙ]:\n{context}\n\nОТВЕЧАЙ СТРОГО НА ОСНОВЕ ТЕКСТА ВЫШЕ."
        print(f"[RAG] Найден контекст, обогащаю промпт!")
    else:
        print(f"[RAG] Совпадений в базе не найдено, отвечаю на общих основаниях.")

    try:
        response = await client.chat.completions.create(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": DYNAMIC_PROMPT},
                {"role": "user", "content": user_text}
            ],
            temperature=0.2, 
            top_p=0.9,         
            max_tokens=200  
        )
        
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        
        answer = response.choices[0].message.content.strip()
        
        print(f"=========================================")
        print(f"[LLM TIMER]: Ответ сгенерирован за {execution_time} сек.")
        print(f"[LLM ОТВЕТ]: {answer}")
        print(f"=========================================\n")
        
        return answer, context
        
    except Exception as e:
        print(f"\n[LLM CRITICAL ERROR]: {e}\n", file=sys.stderr)
        return "Система недоступна.", ""