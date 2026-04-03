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

class Model:
    def __init__(self, *args, **kwargs):
        """В этом методе происходит инициализация модели (загрузка из файла, подготовка к работе)
        args, kwargs могут быть параметры для модели, пока не знаю какие"""
        object.__init__(self)

    def predict(self, message_text: str) -> ModelResponse:
        """Получает сообщение пользователя и выдаёт ответ, метод возвращает объект с аттрибутами priority и category"""
        pass
