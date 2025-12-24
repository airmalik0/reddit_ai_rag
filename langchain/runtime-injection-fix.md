# Решение проблемы Runtime Injection в LangChain

## Проблема

При попытке использовать `ToolRuntime` для доступа к контексту пользователя в инструментах агента возникала ошибка:

```
TypeError: update_cart() missing 1 required positional argument: 'runtime'
TypeError: update_field() missing 1 required keyword-only argument: 'runtime'
```

LangChain не инжектировал параметр `runtime` автоматически, хотя все было настроено по документации.

## Причина

Проблема возникала из-за использования параметра `args_schema` в декораторе `@tool`. Когда вы предоставляете `args_schema`, LangChain использует только эту Pydantic-модель для определения параметров инструмента и **игнорирует сигнатуру функции**.

### Что не работало:

```python
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime

class UpdateCartInput(BaseModel):
    action: str = Field(description="Действие")
    items: List[Dict[str, Any]] = Field(description="Товары")

@tool(args_schema=UpdateCartInput)  # ❌ Проблема здесь
def update_cart(
    action: str,
    items: List[Dict[str, Any]],
    runtime: ToolRuntime[StrawberryContext]  # Не обнаруживается!
) -> Dict[str, Any]:
    user_id = runtime.context.user_id
    # ...
```

LangChain проверял только `UpdateCartInput` и не видел параметр `runtime` в сигнатуре функции.

## Попытки решения

### 1. Изменение типа аннотации ❌
Пробовали менять `"ToolRuntime[StrawberryContext]"` → `ToolRuntime[StrawberryContext]` → `ToolRuntime`
- **Результат:** Не помогло, так как корень проблемы был в `args_schema`

### 2. TYPE_CHECKING imports ❌
Пробовали убрать импорты из блока `TYPE_CHECKING`
- **Результат:** Циркулярный импорт между `agent.py` и `tools.py`

### 3. Создание отдельных context модулей ✅
Создали отдельные файлы для контекста:
- `agent/strawberry/context.py`
- `agent/profile/context.py`
- **Результат:** Решило циркулярные импорты, но основная проблема осталась

## Решение

**Убрать `args_schema` из всех инструментов, которые используют `ToolRuntime`.**

### До:

```python
class UpdateCartInput(BaseModel):
    action: str = Field(description="Действие: add или delete")
    items: List[CartItem] = Field(description="Список товаров")

@tool(args_schema=UpdateCartInput)  # ❌ Удалить это
def update_cart(
    action: str,
    items: List[Dict[str, Any]],
    runtime: ToolRuntime[StrawberryContext]
) -> Dict[str, Any]:
    # ...
```

### После:

```python
@tool  # ✅ Без args_schema
def update_cart(
    action: str,
    items: List[Dict[str, Any]],
    runtime: ToolRuntime[StrawberryContext]
) -> Dict[str, Any]:
    """Обновить корзину покупок: добавить или удалить товары.

    Args:
        action: Действие: add - добавить товар, delete - удалить товар
        items: Список товаров с количеством
    """
    user_id = runtime.context.user_id  # ✅ Работает!
    # ...
```

## Как это работает

Когда вы НЕ предоставляете `args_schema`:

1. LangChain автоматически генерирует Pydantic-схему из сигнатуры функции
2. При генерации схемы LangChain **обнаруживает** параметры типа `ToolRuntime`
3. Эти параметры помечаются как "injected" и исключаются из JSON-схемы для LLM
4. При вызове инструмента `runtime` инжектируется автоматически

### Проверка работы:

```python
from langchain.tools.tool_node import _get_runtime_arg

runtime_arg = _get_runtime_arg(update_cart)
print(runtime_arg)  # "runtime" ✅
```

## Структура контекста

Для избежания циркулярных импортов, контекст вынесен в отдельные файлы:

```
agent/
├── strawberry/
│   ├── context.py      # StrawberryContext
│   ├── agent.py        # imports from context
│   └── tools.py        # imports from context
└── profile/
    ├── context.py      # ProfileContext
    ├── agent.py        # imports from context
    └── tools.py        # imports from context
```

### agent/strawberry/context.py:
```python
from dataclasses import dataclass

@dataclass
class StrawberryContext:
    """Runtime context for Strawberry agent - static configuration."""
    user_id: str
```

## Использование в инструментах

```python
from langchain.tools import tool, ToolRuntime
from .context import StrawberryContext

@tool
def update_cart(
    action: str,
    items: List[Dict[str, Any]],
    runtime: ToolRuntime[StrawberryContext]
) -> Dict[str, Any]:
    """Обновить корзину."""
    # Доступ к контексту
    user_id = runtime.context.user_id

    # Доступ к хранилищу (если настроен)
    store = runtime.store

    # Стрим-райтер для custom events
    writer = runtime.stream_writer

    # Работа с корзиной для конкретного пользователя
    cart = _user_carts.get(user_id, {})
    # ...
```

## Использование агента

```python
from agent.strawberry.agent import build_agent
from agent.strawberry.context import StrawberryContext

agent = build_agent()

result = agent.invoke(
    {"messages": [{"role": "user", "content": "добавь товар"}]},
    context=StrawberryContext(user_id="user_123")  # Передаем контекст
)
```

## Ограничения решения

### Что потеряли:
- **Pydantic валидация** на уровне args_schema
- Не можем использовать сложные модели с дополнительной валидацией (например, `ge=1` для quantity)

### Что осталось:
- ✅ LangChain все равно генерирует JSON-схему из типов функции
- ✅ LLM получает описание параметров из docstring
- ✅ Базовая валидация типов работает
- ✅ Runtime injection работает корректно

### Альтернатива (если нужна строгая валидация):
Можно добавить ручную валидацию внутри функции:

```python
@tool
def update_cart(
    action: str,
    items: List[Dict[str, Any]],
    runtime: ToolRuntime[StrawberryContext]
) -> Dict[str, Any]:
    """Обновить корзину."""
    # Ручная валидация
    if action not in ["add", "delete"]:
        return {"status": "error", "message": "action must be 'add' or 'delete'"}

    for item in items:
        if "product_id" not in item:
            return {"status": "error", "message": "product_id required"}
        if "quantity" not in item or item["quantity"] < 1:
            return {"status": "error", "message": "quantity must be >= 1"}

    user_id = runtime.context.user_id
    # Основная логика...
```

### Обёртка с args_schema + ручной runtime
Если нужна строгая Pydantic-валидация и при этом требуется `ToolRuntime`, можно вынести валидацию в обёрточный инструмент и вручную передавать runtime во внутреннюю функцию:

```python
class CartInput(BaseModel):
    action: Literal["add", "delete"]
    items: list[CartItem]

@tool(args_schema=CartInput)
def validated_update_cart(action: str, items: list) -> dict:
    input_data = CartInput(action=action, items=items)  # Валидация
    return _real_update_cart(input_data, runtime=get_runtime())  # Ручной runtime
```

## Выводы

1. **Не используйте `args_schema` с `ToolRuntime`** - это основная причина проблемы
2. **Выносите Context в отдельные файлы** - избегайте циркулярных импортов
3. **LangChain все равно генерирует схему** - из типов функции и docstring
4. **Добавляйте ручную валидацию при необходимости** - внутри функции

## Дополнительные ресурсы

- [Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [Runtime](https://docs.langchain.com/oss/python/langchain/runtime)
- [Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
