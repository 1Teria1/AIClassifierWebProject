from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    SPAM = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4

class Category(Enum):
    PAYMENT = 1
    DELIVERY = 2
    TECH = 3
    PRODUCT = 4
    SPAM = 5

@dataclass
class ModelResponse:
    priority: Priority
    category: Category
    confidence: float

class Model:
    def __init__(self, *args, **kwargs):
        """В этом методе происходит инициализация модели (загрузка из файла, подготовка к работе)
        args, kwargs могут быть параметры для модели, пока не знаю какие"""
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
