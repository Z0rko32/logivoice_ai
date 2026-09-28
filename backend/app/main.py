from fastapi import FastAPI, UploadFile, File
from app.stt import transcribe_audio
from app.llm import generate_response 
import shutil
import os
import time 

app = FastAPI(title="LogiVoice AI Backend")

@app.post("/api/transcribe")
async def process_audio(file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Этап 1: Распознавание речи (Уши)
        print("[STT] Начинаю распознавание аудио...")
        stt_start = time.time()  # Засекаем время старта STT
        
        user_text = transcribe_audio(file_path)
        
        stt_end = time.time()    # Засекаем время финиша STT
        stt_time = round(stt_end - stt_start, 2) # Считаем разницу
        
        print(f"[STT TIMER]: Текст распознан за {stt_time} сек.")
        print(f"[USER]: {user_text}")
        
        # Если Whisper ничего не расслышал, не дергаем LLM
        if not user_text or user_text.startswith("Ошибка"):
            return {
                "success": False,
                "user_text": "",
                "ai_response": "Я ничего не услышал.",
                "stt_time_sec": stt_time,
                "llm_time_sec": 0.0
            }

        # Этап 2: Генерация ответа (Мозг + RAG)
        print("[LLM] Начинаю генерацию ответа...")
        llm_start = time.time()
        
        ai_response, rag_context = await generate_response(user_text)
        
        llm_end = time.time()
        llm_time = round(llm_end - llm_start, 2)
        
        print(f"[LLM TIMER]: Ответ сгенерирован за {llm_time} сек.")
        print(f"[ИИ]: {ai_response}\n")
        
        total_time = round(stt_time + llm_time, 2)
        print(f"[TOTAL PIPELINE TIME]: {total_time} сек.\n")
        
        return {
            "success": True,
            "user_text": user_text,
            "ai_response": ai_response,
            "rag_context": rag_context,      # Найденный релевантный регламент из базы знаний
            "stt_time_sec": stt_time,        # Время работы Whisper
            "llm_time_sec": llm_time,        # Время генерации Llama
            "total_time_sec": total_time     # Полный цикл
        }
        
    finally:
        # Обязательно удаляем аудиофайл, чтобы не засорять диск
        if os.path.exists(file_path):
            os.remove(file_path)