import os
import sys
import ctranslate2

# ==================== БУЛЛЕТПРУФ CUDA ФИКС ДЛЯ WINDOWS ====================
if sys.platform == "win32":
    try:
        import nvidia.cublas
        import nvidia.cudnn
        
        # Используем __path__, так как __file__ для этих пакетов возвращает None
        cublas_dir = nvidia.cublas.__path__[0]
        cudnn_dir = nvidia.cudnn.__path__[0]
        
        cublas_bin = os.path.join(cublas_dir, "bin")
        cudnn_bin = os.path.join(cudnn_dir, "bin")
        
        if os.path.exists(cublas_bin) and os.path.exists(cudnn_bin):
            # 1. Добавляем в PATH (критично для C++ движка ctranslate2)
            os.environ["PATH"] = cublas_bin + os.path.pathsep + cudnn_bin + os.path.pathsep + os.environ["PATH"]
            # 2. Регистрируем директории DLL в текущем процессе Python
            os.add_dll_directory(cublas_bin)
            os.add_dll_directory(cudnn_bin)
            print("[CUDA FIX] Библиотеки CUDA успешно обнаружены и подключены!")
        else:
            print("[CUDA FIX] Предупреждение: папки bin не найдены внутри пакетов.")
    except Exception as e:
        print(f"[CUDA FIX] Ошибка автоматического поиска либ: {e}")
# ==========================================================================

from faster_whisper import WhisperModel

# МЕНЯЕМ НА TURBO: идеальный баланс VRAM, дикой скорости и точности
MODEL_SIZE = "medium" 

print(f"Инициализация STT пайплайна ({MODEL_SIZE}). Разогреваем тензорные ядра...")

# Оптимизация 1: Аппаратное ускорение и квантование
has_cuda = ctranslate2.get_cuda_device_count() > 0
device = "cuda" if has_cuda else "cpu"
compute_type = "int8_float16" if has_cuda else "int8"

stt_model = WhisperModel(
    MODEL_SIZE, 
    device=device, 
    compute_type=compute_type 
)
print(f"STT модель успешно загружена на {device.upper()} (compute_type={compute_type})!")

def transcribe_audio(file_path: str) -> str:
    """
    Транскрибация аудио c применением VAD-фильтрации и оптимизацией инференса.
    """
    if not os.path.exists(file_path):
        return "Ошибка: Файл не найден"
    
    # Оптимизация 2 и 3: VAD и защита от галлюцинаций
    segments, info = stt_model.transcribe(
        file_path, 
        language="ru",
        beam_size=5, 
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=500), 
        condition_on_previous_text=False 
    )
    
    # Собираем генератор в строку
    text = " ".join([segment.text for segment in segments])
    return text.strip()