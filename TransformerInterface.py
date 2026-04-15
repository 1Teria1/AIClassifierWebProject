from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class Category(Enum):
    PAYMENT = 0
    DELIVERY = 1
    TECH = 2
    PRODUCT = 3
    SPAM = 4


@dataclass
class ModelResponse:
    priority: Priority
    category: Category
    confidence: float


class Model:
    def __init__(self, *args, **kwargs):
        """В этом методе происходит инициализация модели (загрузка из файла, подготовка к работе)
        args, kwargs могут быть параметры для модели"""
        object.__init__(self)

    def predict(self, message_text: str) -> ModelResponse:
        """Получает сообщение пользователя и выдаёт ответ, метод возвращает объект с аттрибутами priority и category"""
        return ModelResponse(Priority(2), Category(1), 0.5)  # Бессмысленный ответ


# Пример использования
if __name__ == "__main__":
    model = Model("Некоторые параметры (или их отсутствие)")
    response = model.predict("Здравствуйте, заказывал у вас товар месяц назад, всё ещё не доставили")
    print(response.category)
    print(response.priority)
    print(response.confidence)
