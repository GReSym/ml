# GReSym ML

ML-сервис дипломного проекта **GReSym (Game Recommendation System)** — рекомендательной системы видеоигр.

Сервис превращает текстовые описания игр в **embedding-векторы**, хранит их в векторной базе данных **Milvus**
и на их основе формирует персональные рекомендации по оценкам пользователя.

> Этот репозиторий содержит только ML-часть GReSym. Пользователи, игры, оценки и HTTP API для клиентов находятся
> в репозитории backend (ASP.NET), который обращается к ML-сервису.

## Возможности

- построение векторов игр моделью [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (работает на CPU);
- хранение векторов в Milvus (индекс HNSW, косинусное сходство);
- массовая векторизация игр из командной строки (с продолжением после прерывания);
- фоновая векторизация по запросам backend через RabbitMQ;
- персональные рекомендации по оценкам пользователя (HTTP API);
- поиск похожих игр.

## Архитектура

```text
           MariaDB (только чтение)
                │
                │ название + описание игры
                ▼
RabbitMQ ──▶ ML worker ──▶ Qwen3-Embedding ──▶ Milvus
  ▲          (векторизация)                      │
  │ {gameId}                                     │ поиск ближайших
  │                                              ▼
Backend (ASP.NET) ──── HTTP POST /recommendations ──▶ ML API
```

- **ML worker** получает из очереди `ml.vectorize-game` идентификатор игры, читает её из MariaDB, строит вектор
  и сохраняет его в Milvus. Если игры в БД больше нет, её вектор удаляется.
- **ML API** строит «профиль вкуса» пользователя по векторам оценённых игр и ищет ближайшие к нему игры в Milvus.
  Модель при этом не загружается, поэтому API запускается за несколько секунд.
- ML-сервис **только читает** MariaDB. Записывает в базу данных только backend.

### Текст для векторизации

Вектор игры складывается из двух частей:

- **описание**: текст `название + описание` (HTML удаляется);
- **отзывы**: до 8 отзывов, отобранных по полезности (оценка Steam), длиной не менее 150 символов, с долей отрицательных,
  соответствующей реальному соотношению отзывов игры в Steam. Каждый отзыв векторизуется отдельно, затем векторы усредняются.

Итоговый вектор: `normalize(v_описание + α · v_отзывы)`, где вес `α` растёт с числом отзывов (до 0,35 при 8 отзывах).
У игр без отзывов вектор совпадает с вектором описания. Обе части хранятся в Milvus отдельно.

При изменении шаблона текста или правил отбора отзывов нужно увеличить `TEXT_VERSION` в `src/ml/text.py`
и заново векторизовать игры.

### Устаревшие векторы

Векторы не пересчитываются автоматически. Вместе с вектором в Milvus хранится «снимок» исходных данных
(хэш описания, id использованных отзывов, число и доля положительных отзывов в Steam). Команда `ml stale` (и `GET /vectorization/stale`)
сравнивает его с текущим состоянием MariaDB и выдаёт список игр, которые стоит векторизовать заново, с причинами:
изменилось описание, изменился набор отбираемых отзывов, сдвинулась доля положительных отзывов, сильно выросло число отзывов,
устарела версия текста, вектора нет, игра удалена. Администратор выбирает игры и отправляет их на векторизацию.

### Алгоритм рекомендаций

1. Каждой оценке сопоставляется вес: оценка 1–10 переводится в диапазон [−1, 1] (10 → 1, 1 → −1),
   при отсутствии оценки используется признак «рекомендую» (±1).
2. Профиль пользователя = нормированная взвешенная сумма векторов понравившихся игр минус половина
   нормированной суммы векторов непонравившихся.
3. В Milvus ищутся игры, ближайшие к профилю по косинусному сходству, кроме уже оценённых.

Если у пользователя нет понравившихся игр с векторами, сервис возвращает пустой список, и backend показывает популярные игры.

## Структура репозитория

```text
.
├── src/ml/
│   ├── __init__.py          # CLI: ml api | worker | vectorize | stale | evaluate | export | import
│   ├── config.py            # Загрузка конфигурации
│   ├── backup.py            # Резервная копия векторов (ml export / ml import)
│   ├── db.py                # Чтение игр и отзывов из MariaDB
│   ├── evaluation.py        # Офлайн-оценка рекомендаций (leave-k-out, базовые стратегии, метрики)
│   ├── text.py              # Очистка HTML, шаблон текста игры, хэш текста
│   ├── reviews.py           # Очистка и отбор отзывов, вес отзывов
│   ├── embedder.py          # Модель эмбеддингов (ONNX Runtime через FastEmbed или Sentence Transformers)
│   ├── vectors.py           # Коллекция Milvus: запись и поиск
│   ├── vectorize.py         # Векторизация игр (в том числе массовая)
│   ├── staleness.py         # Поиск устаревших векторов
│   ├── recommend.py         # Алгоритм рекомендаций
│   ├── api.py               # HTTP API (FastAPI)
│   └── worker.py            # Обработчик очереди RabbitMQ
├── Qwen3-Embedding/         # Файлы модели для torch (не в git, скачиваются отдельно)
├── Qwen3-Embedding-onnx/    # ONNX-версия модели, используется по умолчанию (не в git)
├── config.example.json      # Пример конфигурации (в git)
├── config.json              # Локальная конфигурация с секретами (не в git)
├── Dockerfile               # Образ для `ml api` и `ml worker` (собирается из ../deploy)
├── pyproject.toml
└── uv.lock
```

## Требования

- Linux;
- [uv](https://docs.astral.sh/uv/) 0.12+ (Python 3.14 uv установит сам);
- MariaDB с базой данных GReSym (заполняется backend и парсером);
- Milvus 3.0 и RabbitMQ 4.x (Docker Compose из `../deploy`, RabbitMQ нужен только для `worker`);
- около 4 ГБ свободной оперативной памяти для модели.

PyTorch устанавливается в CPU-версии (индекс `pytorch-cpu` в `pyproject.toml`), видеокарта не нужна.
GPU (AMD ROCm или NVIDIA CUDA) включается в Docker настройкой `TORCH_BACKEND` в `../deploy` (см. README организации).
Без Docker нужно заменить torch вручную и запускать через `uv run --no-sync`:

```bash
uv sync
uv pip uninstall torch
uv pip install --prerelease allow --index https://download.pytorch.org/whl/rocm7.14 torch   # или .../cu132
# индекс версионный (rocm7.14 / cu132) и меняется с релизами torch/ROCm — сверяйтесь с https://download.pytorch.org/whl/
# в config.json: "model": {"device": "cuda", "dtype": "bfloat16"}  (для ROCm тоже cuda; на RX 6600 быстрее float16)
uv run --no-sync ml worker
```

## Установка

### 1. Зависимости

```bash
uv sync
```

### 2. Модель

По умолчанию (`model.backend: onnx`) модель работает через ONNX Runtime на CPU. Её нужно скачать в каталог `Qwen3-Embedding-onnx/`:

```bash
uv run hf download Qdrant/Qwen3-Embedding-0.6B-onnx --include "onnx/model.onnx*" --include "*.json" --local-dir Qwen3-Embedding-onnx
```

Для работы на GPU (`model.backend: torch`) нужна исходная модель в каталоге `Qwen3-Embedding/`:

```bash
uv run hf download Qwen/Qwen3-Embedding-0.6B --local-dir Qwen3-Embedding
```

или через git (нужен `git-lfs`):

```bash
git clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B Qwen3-Embedding
```

### 3. Milvus и RabbitMQ

Milvus 3.0.2, RabbitMQ 4.2 и MariaDB запускаются через Docker Compose из `../deploy`:

```bash
../deploy/manage.sh up
```

Milvus доступен на `localhost:19530`, RabbitMQ на `localhost:5672`. Проверка состояния Milvus:

```bash
curl localhost:9091/healthz
```

Exchange и очереди RabbitMQ создаются автоматически при запуске worker или при первой публикации из backend.

### 4. Конфигурация

Настройки читаются из `config.example.json` (в git), поверх которого накладывается `config.json` (не в git).
Поэтому в `config.json` достаточно указать только секреты:

```json
{
  "database": { "user": "...", "password": "..." },
  "rabbitmq": { "user": "...", "password": "..." }
}
```

Для ML-сервиса достаточно пользователя MariaDB с правами только на чтение (`SELECT`).
Незаполненные значения (`""`, `xxx`, `###`) приводят к ошибке при запуске с указанием, что нужно заполнить.
Путь к конфигурации можно переопределить переменной окружения `GRESYM_ML_CONFIG`.
Отдельные значения переопределяются переменными `GRESYM_ML__<РАЗДЕЛ>__<КЛЮЧ>` (например, `GRESYM_ML__DATABASE__HOST=mariadb`).
Так настраиваются Docker-контейнеры. Если секреты заданы переменными, `config.json` не обязателен.

Основные параметры `config.example.json`:

| Параметр | Значение по умолчанию | Описание |
|---|---|---|
| `database.*` | `localhost:3306`, `GamesRecommend` | Подключение к MariaDB |
| `milvus.uri` | `http://localhost:19530` | Адрес Milvus |
| `milvus.collection` | `games_qwen3_06b_v2` | Коллекция с векторами игр |
| `rabbitmq.*` | `localhost:5672`, `gresym.tasks`, `ml.vectorize-game` | Подключение и имена exchange/очередей (должны совпадать с backend) |
| `rabbitmq.delivery_limit` | `5` | Число попыток обработки сообщения до перемещения в `ml.vectorize-game.dlq` |
| `model.path` | `Qwen3-Embedding` | Путь к модели (относительно каталога репозитория) |
| `model.dimension` | `1024` | Размерность векторов |
| `model.max_seq_length` | `1024` | Максимальная длина текста в токенах (длинные тексты обрезаются) |
| `model.batch_size` | `1` | Размер батча (на CPU 1 оказался быстрее 8) |
| `model.threads` | `null` | Число потоков PyTorch (`null` — по умолчанию) |
| `model.device` | `cpu` | `cpu`, `cuda` (NVIDIA и AMD ROCm) или `auto` |
| `model.dtype` | `float32` | `float32` для CPU, `bfloat16`/`float16` для GPU |
| `reviews.max_selected` | `8` | Сколько отзывов учитывается в векторе игры |
| `reviews.min_chars`, `reviews.max_chars` | `150`, `600` | Минимальная длина отзыва и длина, до которой он обрезается |
| `reviews.weight_max` | `0.35` | Вес отзывов при `max_selected` отзывах |
| `staleness.*` | см. `config.example.json` | Пороги для поиска устаревших векторов |
| `api.host`, `api.port` | `127.0.0.1:8000` | Адрес HTTP API |

### Важно

`config.json` содержит учётные данные и не должен попадать в репозиторий. При смене модели или размерности векторов
нужно указать новое имя коллекции (`milvus.collection`) и заново векторизовать игры.

## Запуск

Все команды выполняются из корня репозитория.

### Массовая векторизация

```bash
uv run ml vectorize --limit 100            # первые 100 игр по id
uv run ml vectorize --ids 1,5,32           # конкретные игры
uv run ml vectorize --all --skip-existing  # все игры, пропуская уже векторизованные
```

Векторы сохраняются порциями по 32 игры, поэтому прерванный запуск можно продолжить с флагом `--skip-existing`.
На CPU скорость около 0,6 игры в секунду без отзывов: 100 игр — около 3 минут, вся база (~67 тыс. игр) — более 30 часов.
Игры с отзывами векторизуются в 2–3 раза дольше.

### Устаревшие векторы

```bash
uv run ml stale                                   # список игр, которые стоит векторизовать заново
uv run ml stale --reason description --limit 50   # только с изменённым описанием
uv run ml vectorize --ids $(uv run ml stale --ids-only --reason reviews)
```

### Резервная копия векторов

```bash
uv run ml export                         # вся коллекция → milvus-<коллекция>-<дата>.npz (модель не нужна)
uv run ml export --out backup.npz
uv run ml import backup.npz              # создаёт коллекцию и загружает векторы
uv run ml import backup.npz --collection test --yes   # в другую коллекцию / в непустую коллекцию
```

Файл содержит все поля, включая векторы, и не зависит от версии Milvus. Импорт проверяет модель и размерность.

### Офлайн-оценка рекомендаций

```bash
uv run python scripts/collect_steam_users.py --seeds <профиль> --count 200   # сбор анонимизированных пользователей Steam (через списки друзей) → data/steam-users.json
uv run ml evaluate                                          # Precision/Recall/HitRate/nDCG@K, MRR против popular и random
uv run ml evaluate --test-selection random-liked --json report.json
```

Для сбора нужны `steam.api_key` и `steam.anonymization_salt` в `config.json`. Файл импортируется в БД через backend
(`POST /api/admin/evaluation/steam-users/import`), а запуск через API (`POST /api/admin/evaluation/runs`) обрабатывает worker.
Подробности: `docs/evaluation.md` в корне проекта.

### Docker

Docker Compose для всего проекта находится в `../deploy` (MariaDB, RabbitMQ, Milvus, а в профиле `app` ещё API, `ml api` и `ml worker`).
Образ собирается из `Dockerfile` в этом репозитории. Модель в образ не входит: каталоги `Qwen3-Embedding/` и `Qwen3-Embedding-onnx/` монтируются только для чтения.

### Worker (очередь векторизации)

```bash
uv run ml worker
```

Загружает модель и обрабатывает сообщения из очереди `ml.vectorize-game`. Формат сообщения:

```json
{ "gameId": 123, "reason": "game-updated" }
```

Некорректные сообщения сразу перемещаются в `ml.vectorize-game.dlq`. При ошибке обработки сообщение возвращается
в очередь и после 5 попыток тоже попадает в `ml.vectorize-game.dlq`.

### HTTP API

```bash
uv run ml api
```

Документация Swagger доступна по адресу `http://127.0.0.1:8000/docs`.

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/health` | Состояние сервиса и количество векторов |
| `POST` | `/recommendations` | Рекомендации по оценкам пользователя |
| `GET` | `/games/{id}/similar?limit=10` | Похожие игры (404, если у игры нет вектора) |
| `GET` | `/vectorization/stale?reason=…&limit=100&offset=0` | Игры с устаревшими векторами, по убыванию приоритета |

Пример запроса рекомендаций:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H 'Content-Type: application/json' \
  -d '{"ratings": [{"gameId": 1, "rating": 10}, {"gameId": 6, "recommend": false}], "limit": 5}'
```

Ответ:

```json
{ "items": [{ "gameId": 26, "score": 0.47 }, { "gameId": 16, "score": 0.44 }], "usedRatings": 2 }
```

`rating` — оценка от 1 до 10, `recommend` — признак «рекомендую»; должно быть указано хотя бы одно из них.
`score` — косинусное сходство с профилем пользователя (чем больше, тем лучше).

API не имеет аутентификации и по умолчанию слушает только `127.0.0.1`. Его вызывает backend, а не клиенты напрямую.

## Модель

Используется **Qwen3-Embedding-0.6B** через Sentence Transformers:

- 0,6 млрд параметров, контекст до 32 тыс. токенов, векторы размерности 1024;
- поддержка более 100 языков, включая русский (описания игр в базе данных на русском);
- описания игр векторизуются без инструкции (как документы).

Модель хранится в формате bfloat16, но загружается в float32: на процессорах без аппаратной поддержки bf16
(например, AMD Ryzen 5 5500U) это примерно в 4 раза быстрее. В памяти модель занимает около 2,4 ГБ.
