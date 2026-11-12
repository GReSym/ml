# GReSym ML

ML-сервис дипломного проекта **GReSym (Game Recommendation System)** — рекомендательной системы видеоигр.

Сервис превращает текстовые описания игр в **embedding-векторы**, хранит их в векторной базе данных **Milvus**
и на их основе формирует персональные рекомендации по оценкам пользователя.

> Этот репозиторий содержит только ML-часть GReSym. Пользователи, игры, оценки и HTTP API для клиентов находятся
> в репозитории backend (ASP.NET), который обращается к ML-сервису.

## Возможности

- построение векторов игр моделью [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m) (по умолчанию использует ONNX Runtime на CPU);
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
RabbitMQ ──▶ ML worker ──▶ EmbeddingGemma ──▶ Milvus
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
- **отзывы**: до 8 отзывов, отобранных по полезности (оценка Steam), с долей отрицательных, соответствующей реальному
  соотношению отзывов игры в Steam. Каждый отзыв векторизуется отдельно, затем векторы усредняются.

Итоговый вектор (`reviews.fusion = rcenter-concat`, с 2026-11-12): `normalize([v_описание, √w · normalize(v_отзывы − r̄)])`,
где `w = 0,7 · min(1, отзывов / 8)`, а `r̄` — средний вектор отзывов каталога, зафиксированный в файле `reviews.mean_path`.
Размерность вектора поиска 1536 (две половины по 768). Косинус двух игр = сходство описаний + сходство центрированных отзывов.
У игр без отзывов вторая половина нулевая. Обе части хранятся в Milvus отдельно, поэтому способ объединения и вес меняются
без модели: `ml recombine`. Почему так: `docs/evaluation.md` («Fusion sweep») и `docs/ml.md` («Fusion»).

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
│   ├── __init__.py          # CLI: ml api | worker | vectorize | index-text | stale | evaluate | judge | recombine | export | import
│   ├── config.py            # Загрузка конфигурации
│   ├── backup.py            # Резервная копия векторов (ml export / ml import)
│   ├── db.py                # Чтение игр и отзывов из MariaDB
│   ├── evaluation.py        # Офлайн-оценка рекомендаций (leave-k-out, базовые стратегии, метрики)
│   ├── text.py              # Очистка HTML, шаблон текста игры, хэш текста
│   ├── reviews.py           # Очистка и отбор отзывов, вес отзывов
│   ├── fusion.py            # Объединение векторов описания и отзывов (sum / rcenter / concat / rcenter-concat)
│   ├── recombine.py         # Пересборка коллекции с другим объединением без модели (ml recombine)
│   ├── memstore.py          # Векторы в памяти и точный поиск для сравнения способов объединения (ml evaluate --fusion)
│   ├── cf.py                # Базовые методы коллаборативной фильтрации для оценки: EASE, item-CF, популярность в сообществе (ml evaluate --cf)
│   ├── judge.py, judged.py  # LLM-судья релевантности и метрики по его оценкам (ml judge)
│   ├── embedder.py          # Модель эмбеддингов (ONNX Runtime через FastEmbed или Sentence Transformers)
│   ├── vectors.py           # Коллекция Milvus: запись и поиск
│   ├── vectorize.py         # Векторизация игр (в том числе массовая)
│   ├── staleness.py         # Поиск устаревших векторов
│   ├── recommend.py         # Алгоритм рекомендаций
│   ├── api.py               # HTTP API (FastAPI)
│   └── worker.py            # Обработчик очереди RabbitMQ
├── models/                  # Модели (не в git): embeddinggemma-300m/ и embeddinggemma-300m-onnx/
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

Модели хранятся в игнорируемом git каталоге `models/`. По умолчанию (`model.backend: onnx`) используется ONNX Runtime на CPU. ONNX-модель нужно скачать в `models/embeddinggemma-300m-onnx/`:

```bash
uv run hf download onnx-community/embeddinggemma-300m-ONNX --local-dir models/embeddinggemma-300m-onnx
```

Для работы на GPU (`model.backend: torch`) нужна исходная модель в `models/embeddinggemma-300m/`:

```bash
uv run hf download google/embeddinggemma-300m --local-dir models/embeddinggemma-300m
```

или через git (нужен `git-lfs`):

```bash
git clone https://huggingface.co/google/embeddinggemma-300m models/embeddinggemma-300m
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
| `milvus.collection` | `games_gemma_03b_rc07` | Коллекция с векторами игр (одна коллекция = одна модель и один `reviews.fusion`) |
| `rabbitmq.*` | `localhost:5672`, `gresym.tasks`, `ml.vectorize-game` | Подключение и имена exchange/очередей (должны совпадать с backend) |
| `rabbitmq.delivery_limit` | `5` | Число попыток обработки сообщения до перемещения в `ml.vectorize-game.dlq` |
| `model.path` | `models/embeddinggemma-300m` | Путь к модели для torch (относительно каталога репозитория) |
| `model.onnx_path` | `models/embeddinggemma-300m-onnx` | Путь к ONNX-модели |
| `model.dimension` | `768` | Размерность векторов |
| `model.query_prefix` | `task: search result | query: ` | Префикс поискового запроса для EmbeddingGemma |
| `model.document_prefix` | `title: none | text: ` | Префикс документов для EmbeddingGemma |
| `model.max_seq_length` | `1024` | Максимальная длина текста в токенах (длинные тексты обрезаются) |
| `model.batch_size` | `1` | Размер батча (на CPU 1 оказался быстрее 8) |
| `model.threads` | `null` | Число потоков PyTorch (`null` — по умолчанию) |
| `model.device` | `cpu` | `cpu`, `cuda` (NVIDIA и AMD ROCm) или `auto` |
| `model.dtype` | `float32` | `float32` для CPU, `bfloat16`/`float16` для GPU |
| `reviews.max_selected` | `8` | Сколько отзывов учитывается в векторе игры |
| `reviews.max_chars` | `600` | Длина, до которой обрезается отзыв |
| `reviews.weight_max` | `0.7` | Вес отзывов при `max_selected` отзывах |
| `reviews.fusion` | `rcenter-concat` | Способ объединения описания и отзывов: `sum`, `rcenter`, `concat`, `rcenter-concat` |
| `reviews.mean_path` | `models/fusion/review-mean-games_gemma_03b_rc07.npy` | Зафиксированный средний вектор отзывов (для `rcenter*`), пишет `ml recombine` |
| `judge.*` | пусто | LLM-судья для `ml judge` (OpenAI-совместимый API; `api_key` — в `config.json`) |
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

Файл содержит все поля, включая векторы и средний вектор отзывов, и не зависит от версии Milvus. Импорт проверяет модель,
размерность и соответствие `reviews.fusion`.

### Другой способ объединения векторов

```bash
uv run ml recombine --from games_gemma_03b_v1 --to games_gemma_03b_rc07   # ~5 минут, модель не нужна
```

Пересчитывает вектор поиска каждой игры по `reviews.fusion` / `reviews.weight_max` из сохранённых частей в новую коллекцию
(исходная только читается). Для `rcenter*` при отсутствии файла `reviews.mean_path` вычисляет и сохраняет средний вектор отзывов.

### Офлайн-оценка рекомендаций

```bash
uv run python scripts/collect_steam_users.py --seeds <профиль> --count 200   # сбор анонимизированных пользователей Steam (через списки друзей) → data/steam-users.json
uv run ml evaluate                                          # Precision/Recall/HitRate/nDCG@K, MRR против popular и random
uv run ml evaluate --test-selection random-liked --json report.json
uv run ml evaluate --center --fusion sum:0.35,rcenter-concat:0.7 --json sweep.json   # сравнение способов объединения, парные CI
uv run ml judge sweep.json --users 100 --json judged.json                           # LLM-судья (нужен judge.* в config.json)
uv run ml evaluate --center --cf --json cf.json                                     # + коллаборативная фильтрация (EASE, item-CF) на тех же пользователях
```

Для сбора нужны `steam.api_key` и `steam.anonymization_salt` в `config.json`. Файл импортируется в БД через backend
(`POST /api/admin/evaluation/steam-users/import`), а запуск через API (`POST /api/admin/evaluation/runs`) обрабатывает worker.
Подробности: `docs/evaluation.md` в корне проекта.

### Docker

Docker Compose для всего проекта находится в `../deploy` (MariaDB, RabbitMQ, Milvus, а в профиле `app` ещё API, `ml api` и `ml worker`).
Образ собирается из `Dockerfile` в этом репозитории. Модель в образ не входит: выбранный каталог модели из `models/` монтируется только для чтения.

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

По умолчанию используется **EmbeddingGemma 300M** через ONNX Runtime (FastEmbed):

- 300 млн параметров, векторы размерности 768;
- поддержка более 100 языков, включая русский (описания игр в базе данных на русском);
- описания игр векторизуются как документы, запросы используют префикс задачи retrieval.

Для GPU можно переключить `model.backend` на `torch`; для этого скачайте исходную модель в каталог `model.path`.
