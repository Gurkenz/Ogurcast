# Ogurcast

Ogurcast - локальный тестовый harness для WhisperX на Windows. Он поднимает минимальный FastAPI-сервер, принимает аудио или видео файл, запускает WhisperX и сохраняет набор файлов, готовых для дальнейшей постобработки.

Проект расположен строго здесь:

```text
Z:\Ogurcast
```

Все управляемые кэши, модели, загрузки, временные файлы, логи и результаты направлены внутрь `Z:\Ogurcast`. Это нужно, чтобы тест не засорял `AppData`, системный temp, default Hugging Face cache и старый `C:\Podcast`. Если сторонняя библиотека проигнорирует эти переменные окружения, это надо проверять отдельно по фактическим путям загрузки.

## Требования

- Windows 10/11.
- Python 3.11.
- FFmpeg в `PATH`.
- NVIDIA GPU рекомендуется.
- Hugging Face token для diarization.
- Принятое model agreement для pyannote-моделей на Hugging Face.

## Установка

```powershell
Z:
cd Z:\Ogurcast
.\scripts\setup_env.ps1
.\scripts\install_deps.ps1
.\scripts\check_env.ps1
```

`requirements.txt` намеренно не содержит `torch`, `torchaudio` и `torchvision`. PyTorch ставится отдельно из CUDA wheel index в `scripts\install_deps.ps1`.

Важно: WhisperX 3.8.6 зависит от `torch~=2.8.0`, поэтому `install_deps.ps1` после установки WhisperX принудительно возвращает CUDA wheels `torch==2.8.0+cu126`, `torchaudio==2.8.0+cu126` и `torchvision==0.23.0+cu126`. Без этого pip может заменить CUDA-сборку на CPU-сборку.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

`pytest` пишет временные файлы в `Z:\Ogurcast\tmp\pytest`, а не в системный temp.

## Warmup

```powershell
.\scripts\warmup_models.ps1
```

Warmup загружает ASR-модель, alignment-модель и, если найден `HF_TOKEN`, инициализирует diarization pipeline. Лог пишется в:

```text
Z:\Ogurcast\logs\warmup.log
```

## CLI test

```powershell
.\scripts\run_one.ps1 -InputFile "Z:\Ogurcast\uploads\test.mp3" -OutputDir "Z:\Ogurcast\outputs"
```

## Server

```powershell
.\scripts\run_server.ps1
```

Открыть:

```text
http://127.0.0.1:7860
```

## LLM audit

Основной LLM-этап `diarized_asr_audit` работает через repo-managed `llama-server` OpenAI-compatible API. Приложение не скачивает multi-GB модели во время транскрибации: `llama-server.exe` должен лежать в `Z:\Ogurcast\tools`, а GGUF-модель в `Z:\Ogurcast\models\llm`.

Defaults:

```text
OGURCAST_LLM_PROVIDER=llamacpp
OGURCAST_LLM_BASE_URL=http://127.0.0.1:8080/v1
OGURCAST_LLM_MODEL=qwen3-8b-q4_k_m.gguf
OGURCAST_LLM_TEMPERATURE=0.05
OGURCAST_LLM_TIMEOUT_SEC=180
OGURCAST_LLM_API_KEY=
OGURCAST_LLAMA_SERVER_EXE=Z:\Ogurcast\tools\llama-server.exe
OGURCAST_LLAMA_MODEL_PATH=Z:\Ogurcast\models\llm\qwen3-8b-q4_k_m.gguf
OGURCAST_LLM_AUTO_AUDIT=true
```

После успешной diarization backend автоматически ставит `diarized_asr_audit` в очередь. Если `llama-server` еще не поднят, приложение пытается запустить `tools\llama-server.exe` с моделью из `OGURCAST_LLAMA_MODEL_PATH`, проверяет `/health` и `/v1/models`, затем вызывает `/v1/chat/completions`. Если LLM audit падает, WhisperX job остается `done`, а Workbench показывает failed audit status и audit dir.

В Workbench кнопка `LLM: повторить аудит` запускает ручной rerun stage `diarized_asr_audit` через `POST /api/jobs/{job_id}/llm/runs`. Модель возвращает только structured audit findings. Сервер проецирует их в review-layer artifacts:

- text-safe findings -> `review\correction_suggestions.json`;
- uncertain/audio findings -> `review\audio_flags.json`;
- glossary/entity findings -> `review\entity_annotations.json`;
- diarization/source/technical findings -> `review\audit_suggestions.json` и `review\technical_flags.json`.

Применение текста остается ручным через Review Workbench или через безопасный batch для low-risk ASR corrections.

Developer panel показывает:

- LLM provider, base URL, configured model, доступные модели и loaded-state;
- per-run model/temperature/timeout;
- редактируемый `system instruction`, `user payload template`, описание стадии и schema notes.

Локальные dev-профили сохраняются в:

```text
Z:\Ogurcast\config\llm_profiles.local.json
```

Этот файл исключен из git. Каждый LLM-запуск сохраняет snapshot профиля, request, response, events, prompt hash и metadata в:

```text
review\llm_runs\<run_id>\
```

Старый endpoint `POST /api/jobs/{job_id}/llm/postprocess` оставлен как compatibility wrapper для stage `asr_correction`.

LLM status API:

```text
GET /api/llm/status
GET /api/llm/profiles
PUT /api/llm/profiles/diarized_asr_audit
POST /api/jobs/{job_id}/llm/runs
GET /api/jobs/{job_id}/llm/runs/{run_id}
```

Важно: доступность HTTP endpoint не равна готовности модели. `GET /api/llm/status` должен показывать loaded model evidence: `availableModels` и `modelLoaded=true`. Для `llamacpp` один загруженный model id считается достаточным loaded-state; для LM Studio лучше выставить точный `OGURCAST_LLM_MODEL`.

LM Studio остается fallback: задайте `OGURCAST_LLM_PROVIDER=lmstudio`, `OGURCAST_LLM_BASE_URL=http://127.0.0.1:1234/v1` и фактический model id из `/v1/models`.

User-facing correction queue не содержит фоновые нормализации пробелов и no-op правки. Нормализация whitespace применяется при создании review transcript. Низкая уверенность ASR сохраняется отдельно как audio flags, а не как фиктивная правка вида `слово -> слово`.

## Audio review

Для завершенной job Workbench подключает исходный файл через:

```text
GET /api/jobs/{job_id}/audio
```

Endpoint отдает только исходный файл из `Z:\Ogurcast\uploads`, найденный через `metadata.json`. Произвольные файлы проекта и path traversal отклоняются. В Listen Review доступны play segment, play correction с окном `±3 sec` и подсветка слов по word-level timestamps.

Скорость playback задается в Workbench через dropdown. Text и Final view показывают speaker turns: соседние сегменты одного speaker объединяются в читаемый блок, но исходные segment IDs и timestamps остаются в review artifacts для playback и ручной проверки.

## Entity verification safety

Автоматического web-поиска сущностей сейчас нет. Entity Review использует только локальные regex-кандидаты из transcript, а кнопка `Искать еще` не выполняет внешних запросов. Будущий web lookup должен быть opt-in по явному клику, без передачи raw audio, локальных путей, секретов или полного transcript, и не должен мутировать transcript напрямую.

## Выходные файлы

Каждый запуск создает папку:

```text
Z:\Ogurcast\outputs\<safe_audio_stem>_whisperx_<YYYYMMDD_HHMMSS>
```

Внутри:

```text
result_raw.json
segments.json
words.json
transcript.txt
speaker_transcript.txt
transcript.srt
transcript.vtt
metadata.json
run.log
```

Все JSON пишутся как UTF-8 с `ensure_ascii=False` и `indent=2`.

## Кэши и временные пути

Основные пути:

```text
HF_HOME=Z:\Ogurcast\.cache\huggingface
HF_HUB_CACHE=Z:\Ogurcast\.cache\huggingface\hub
HF_TOKEN_PATH=Z:\Ogurcast\.cache\huggingface\token
TRANSFORMERS_CACHE=Z:\Ogurcast\.cache\huggingface\transformers
TORCH_HOME=Z:\Ogurcast\.cache\torch
PIP_CACHE_DIR=Z:\Ogurcast\.cache\pip
NLTK_DATA=Z:\Ogurcast\.cache\nltk
TMP=Z:\Ogurcast\tmp
TEMP=Z:\Ogurcast\tmp
```

`tools\ffmpeg.cmd` - локальный shim для Windows. Он нужен потому, что в проверенном окружении Python `subprocess.run(["ffmpeg", ...])` падал с `[WinError 5]`, хотя PowerShell запускал внешний `ffmpeg.exe`. Runner декодирует аудио через проверенный абсолютный путь из `OGURCAST_FFMPEG`.

## Troubleshooting

`CUDA недоступна`: проверьте драйвер NVIDIA, CUDA-compatible PyTorch wheel и вывод `nvidia-smi`. На CPU WhisperX может работать, но будет медленно.

`Недостаточно видеопамяти`: уменьшите `batch_size` до `4`, поставьте `compute_type=int8` или выберите модель меньше.

`Ошибка: отсутствует HF_TOKEN`: проверьте `Z:\Ogurcast\.env` или задайте `HF_TOKEN` в текущем процессе.

`Ошибка: нет доступа к модели pyannote`: проверьте токен и принятие model agreement на Hugging Face. Один только токен не помогает, если agreement не принят.

`Ошибка: не найден FFmpeg`: установите FFmpeg и добавьте его в `PATH`.

`[WinError 5] Отказано в доступе` при загрузке аудио: это может быть не файл, а запуск `ffmpeg` из Python. В проекте есть `tools\ffmpeg.cmd`, а runner использует `OGURCAST_FFMPEG`, чтобы обходить хрупкое PATH-разрешение Windows.

Проблемы русского alignment: попробуйте явно оставить `language=ru`, обновить WhisperX и проверить, что alignment-модель скачалась в проектный cache.

NLTK не должен писать в `AppData`: для этого задан `NLTK_DATA=Z:\Ogurcast\.cache\nltk`.

Метки спикеров неидеальны: diarization не гарантирует стабильные имена людей, особенно на перекрывающейся речи, шуме и коротких репликах.

`torchcodec is not installed correctly`: в проверенном окружении pyannote выводит warning о `torchcodec` DLL. Warmup при этом проходит, а runner передает в diarization уже загруженное аудио. Если конкретная версия pyannote начнет декодировать файл напрямую и падать, фиксировать надо совместимость `torch`, `torchcodec` и FFmpeg DLL.

`hf_xet package is not installed`: Hugging Face падает обратно на обычную HTTP-загрузку. Это медленнее, но не ломает работу.

Windows symlink cache warning: без Developer Mode или запуска от администратора Hugging Face cache работает в degraded mode и может занимать больше места. Это не выводит cache за пределы `Z:\Ogurcast`.

## Security

`.env` содержит Hugging Face token и исключен из git через `.gitignore`. Значение токена не выводится в health API, metadata, логи и ответы сервера. После тестирования токен нужно ротировать.
