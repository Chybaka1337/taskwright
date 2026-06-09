# Архитектура фреймворка — ВКР

> Каноническая запись архитектурных решений: это выжимка из предыдущих чатов, **очищенная от метрики
> объёма кода / времени**, дополненная тремя ветками Runtime (из `research_context.md` 4.3), структурой
> проекта и определением `TaskResult`.
>
> Продукт — **Python-библиотека** (pip-пакет), game-agnostic, аналог Hibernate / Spring Data / sklearn:
> подключается как зависимость, данные приносит разработчик. Ценность — закрытие задокументированного
> разрыва: в open-source нет фреймворка с типизированными контрактами на категории аналитических задач
> (`research_context.md` 6.2). **Никаких метрик экономии кода или времени работа не использует.**

## Суть

Фреймворк = **контракты + Runtime** в равной мере. Контракты декларируют ЧТО (типы данных, сигнатуры
методов); Runtime берёт КАК (оркестрация, валидация, метрики, сохранение). Без Runtime контракты
бесполезны; без контрактов Runtime не работает. Архитектурный аналог — Spring Data Repository +
Criteria/Specification. **Новизна:** паттерн «декларативные контракты + оркеструющий рантайм» известен,
домен (game analytics) новый — в OSS этот паттерн к нему не применялся.

## Цель и критерии (зафиксированы)

> «Разработать программный фреймворк, предоставляющий унифицированные типизированные контракты для трёх
> категорий аналитических задач (с обучением, без учителя, детерминированный расчёт) в области игровой
> аналитики, и подтвердить его работоспособность через реализацию эталонных задач каждой категории на
> данных не менее двух различных игр.»

| Показатель | Значение |
|---|---|
| Число типизированных контрактов | 3 (по числу категорий) |
| Число эталонных реализаций задач через контракты | ≥ 3 (не менее одной на категорию) |
| Число разных игровых датасетов в апробации | ≥ 2 |

Качественный критерий: работающий фреймворк с 3 контрактами, работающими эталонными реализациями и
демонстрируемой переносимостью между играми. Работоспособность подтверждается сопоставлением метрик с опубликованными результатами на тех же
публичных датасетах (напр., UCI Dota2 Games Results ≈ 60% точности — DOI 10.24432/C5W593, CC BY 4.0).
Прямое сравнение с данными статей [REF-006] Hodge и
[REF-025] Periáñez невозможно — их датасеты не опубликованы (воспроизводится методология, не
абсолютные числа); частичное прямое сравнение доступно с [REF-007] Yang (публичный sample 100 игр
Honor of Kings на GitHub). Для задач без опубликованного бенчмарка на том же наборе (синтетический
T1; KPI на OpenDota T12) работоспособность подтверждается невырождённостью результата: для обучаемых
категорий — превышение тривиальной базовой линии, для детерминированной — выполнение sanity-предиката
(σ(κ)=1); числовой порог производительности не требуется. Это соответствует формализации valid(j) в §5.4 ВКР.

**Конкретные датасеты апробации** (определены через deep research, см. `vkr_plan.md` раздел «Apробация на разных играх»):
- T4 (основной): UCI Dota2 Games Results (Dota 2, CC BY 4.0, 102 944 игры)
- T4 (трансферабельность): LoL Diamond Ranked Games 10min (League of Legends, ~9 879 игр)
- T2: Cookie Cats (мобильный пазл, 90 189 игроков, retention_1/_7)
- T1: Predict Online Gaming Behavior (синтетический, CC BY 4.0, 40 034 игрока — пометить как синтетический)
- T12: OpenDota player_matches dump small sample (Dota 2, ~4 GB CSV)

## Уровень 1 — контракт телеметрии (Pydantic)

Фиксированная база + расширяемые подклассы на типы событий. НЕ YAML (Python-only, одна точка
редактирования, type-safety, IDE-autocomplete).

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Literal

class BaseEvent(BaseModel):
    """Обязательная база — фиксированный словарь фреймворка."""
    user_id: str
    event_timestamp: datetime
    event_id: str            # idempotency
    event_name: str          # тип события; в подклассах сужается до Literal

class MatchFinished(BaseEvent):
    event_name: Literal["match_finished"] = "match_finished"
    match_id: str
    duration_sec: int
    hero: str
    kills: int
    deaths: int
```

Фиксированное (требует фреймворк): `user_id`, `event_timestamp`, `event_id`, `event_name`.
Расширяемое (объявляет разработчик): любые поля своих событий.

## Уровень 2 — контракты задач (ABC)

Разработчик наследует и реализует декларативные методы; Runtime оркеструет остальное.

```python
class SupervisedTask(ABC):
    @abstractmethod
    def build_features(self, events) -> pd.DataFrame: ...
    @abstractmethod
    def build_labels(self, events) -> pd.Series: ...
    @abstractmethod
    def build_model(self): ...

class UnsupervisedTask(ABC):
    @abstractmethod
    def build_features(self, events) -> pd.DataFrame: ...
    @abstractmethod
    def build_model(self): ...
    @abstractmethod
    def interpret(self, clusters, X) -> ClusterProfile: ...

class DeterministicTask(ABC):
    @abstractmethod
    def aggregate(self, events, window) -> KPI: ...
```

| Контракт | Апробационные задачи |
|---|---|
| `SupervisedTask` | T2 churn, T4 win prediction |
| `UnsupervisedTask` | T1 player segmentation |
| `DeterministicTask` | T12 retention/KPI |

## Уровень 3 — Runtime: три ветки оркестрации

`runtime.run(task, events)` диспетчит по типу контракта. Ветки построены на типовых конвейерах из
`research_context.md` 4.3 и провалидированы против контрактов.

**Общее для всех веток:** валидация `events` по Pydantic-схеме → … → сохранение результата + лог
(JSON Lines) → `TaskResult`.

**Supervised** (T2, T4):
1. Валидация `events` (Pydantic)
2. `build_features` → X, `build_labels` → y
3. Проверка согласованности X/y (число строк, индексы)
4. train/test split (временной — для time-aware задач, например T4)
5. `build_model` → model; `model.fit(X_train, y_train)`
6. `predict(X_test)` → ŷ; метрики (AUC/F1 — классификация, MAE — регрессия)
7. Сохранение model (joblib) + `manifest.json` + лог → `TaskResult`

**Unsupervised** (T1):
1. Валидация `events`
2. `build_features` → X
3. Нормализация X (дефолт Runtime, переопределяема; либо внутри `build_features`)
4. `build_model` → model; `model.fit_transform(X)` → ClusterAssignment
5. Метрики качества кластеров (silhouette / Davies-Bouldin) — без y
6. `interpret(clusters, X)` → ClusterProfile
7. Сохранение + лог → `TaskResult` (ClusterAssignment + ClusterProfile + QualityMetric)

**Deterministic** (T12) — без модели и без train/test:
1. Валидация `events` (Pydantic)
2. Дедупликация + фильтр по временному окну
3. `aggregate(events, window)` → KPI
4. Sanity-валидация результата (например, retention ∈ [0, 1])
5. Сохранение KPI + лог → `TaskResult`

Нюансы валидации: «нормализация» (unsupervised) и «дедупликация + sanity» (deterministic) — это
ответственность Runtime, а не методы контракта; `aggregate` принимает окно как параметр.

## TaskResult (MVP-минимум)

- ссылка на сохранённый артефакт (путь к модели/результату);
- метрики (dict);
- метаданные запуска: timestamp, имя задачи, категория, датасет.

Расширенный набор (feature importance, ROC-данные, confusion matrix) — зона развития, добавляется,
когда конкретная задача этого потребует.

## Технический стек (зафиксирован)

Python 3.11+; Pydantic v2 (контракты/валидация); pytest; scikit-learn (T1), XGBoost (T2, T4),
pandas (T12) — эталонные реализации; joblib + JSON-manifest с timestamp (модели); poetry (сборка);
Sphinx + autodoc (API); `logging` в формате JSON Lines.

## Структура проекта (два репозитория)

Имя пакета — `taskwright`. Структура — **два репозитория**: (1) библиотека и (2) отдельный
проект апробации + demo, который подключает библиотеку как зависимость (аналог Spring PetClinic).

**Репозиторий 1 — библиотека `taskwright`** (src-layout, идёт на PyPI):

```
taskwright/
├── pyproject.toml              # poetry
├── README.md
├── src/taskwright/
│   ├── telemetry/events.py     # BaseEvent + базовая схема
│   ├── tasks/
│   │   ├── supervised.py       # SupervisedTask (ABC)
│   │   ├── unsupervised.py     # UnsupervisedTask (ABC)
│   │   └── deterministic.py    # DeterministicTask (ABC)
│   ├── runtime/runner.py       # run(task, events) + три ветки
│   ├── metrics/                # расчёт метрик по категориям
│   ├── persistence/            # joblib + manifest.json
│   └── result.py               # TaskResult
├── examples/                   # МИНИ-пример на синтетике (для README/quickstart)
├── tests/
└── docs/                       # Sphinx
```

**Репозиторий 2 — апробация + demo `taskwright-demo`** (подключает `taskwright`; имя на выбор):

```
taskwright-demo/
├── pyproject.toml              # зависит от taskwright (в разработке — path/editable)
├── tasks/                      # эталонные реализации контрактов
│   ├── t1_segmentation/        # UnsupervisedTask
│   ├── t2_churn/               # SupervisedTask
│   ├── t4_win_prediction/      # SupervisedTask
│   └── t12_retention/          # DeterministicTask
├── data/                       # загрузка/подготовка датасетов (большие — в .gitignore)
├── runs/                       # TaskResult + материал для графиков (см. ниже)
├── figures/                    # экспорт рисунков для §7–8 ВКР
├── demo/                       # Streamlit-инструмент визуализации
└── README.md
```

Прогоны апробации сохраняют, помимо TaskResult, материал для графиков: для supervised — `y_true`/`y_score`
теста (ROC, матрица ошибок); для unsupervised — назначения кластеров + 2D-проекция; для deterministic —
ряд retention. Отсюда строятся рисунки §7–8 и работает demo. MVP-`TaskResult` библиотеки это НЕ
расширяет — это артефакты прогонов в репозитории апробации, а не поля результата в API библиотеки.

## Границы scope

**ДЕЛАЕТ:** декларирует контракты (Pydantic + ABC); валидирует телеметрию; оркеструет пайплайн по
категориям; стандартизирует метрики и сохранение; логирует.

**НЕ делает:** не выбирает метрику (P8) и алгоритм (P9) — это разработчик; не поднимает HTTP-сервер;
не строит дашборды; не работает с конкретной БД (разработчик передаёт DataFrame); не содержит готовых
моделей (только контракты); не решает организационные проблемы (P13).

## Demo для защиты

Отдельный артефакт поверх фреймворка (аналог Spring PetClinic), НЕ часть библиотеки; живёт в
репозитории апробации (`taskwright-demo`). Стек — **Streamlit**: чистый Python, быстрый путь к
скринам и живому демо, без отдельного фронтенд-стека. Инструмент потребляет TaskResult и
сохранённый материал прогонов и визуализирует результаты апробации (ROC, матрица ошибок, scatter
кластеров, кривые retention) — скрины идут в презентацию и, как визуализация результатов, в §7–8.

## Зоны развития (не в MVP)

- Hooks Runtime (`before_fit` / `after_evaluate`)
- Расширенный `TaskResult` (feature importance, ROC, confusion matrix)
- MLflow-интеграция для версионирования
- RL-категория (T7/T8/T9) — требует симуляторов / интерактивных сред
