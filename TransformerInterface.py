from dataclasses import dataclass
from enum import Enum
import torch
from transformers import AutoModel, AutoTokenizer


# Класс двухголовой модели
class RuBERTMultiTask(torch.nn.Module):
    def __init__(self, model_name, num_categories=5, num_priorities=4, *args, **kwargs):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # Две независимые головы
        self.category_head = torch.nn.Linear(hidden_size, num_categories)
        self.priority_head = torch.nn.Linear(hidden_size, num_priorities)

    def forward(self, input_ids, attention_mask, token_type_ids=None, *args, **kwargs):
        # Общий кодировщик
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Берем CLS-токен (вектор всего предложения)
        pooled = outputs.last_hidden_state[:, 0, :]

        # Два выхода
        category_logits = self.category_head(pooled)
        priority_logits = self.priority_head(pooled)

        return category_logits, priority_logits


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
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.model = RuBERTMultiTask("DeepPavlov/rubert-base-cased")
        self.state_dict = torch.load("model.pt", map_location=torch.device('cpu'))
        self.model.load_state_dict(self.state_dict)

    def predict(self, message_text: str) -> ModelResponse:
        """Получает сообщение пользователя и выдаёт ответ, метод возвращает объект с аттрибутами priority и category"""
        tokenized = self.tokenizer(
            message_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            category_logits, priority_logits = self.model(**tokenized)
        category_probs = torch.softmax(category_logits, dim=-1)
        priority_probs = torch.softmax(priority_logits, dim=-1)
        category_pred = torch.argmax(category_probs, dim=-1).item()
        priority_pred = torch.argmax(priority_probs, dim=-1).item()
        category_confidence = category_probs[0, category_pred].item()
        return ModelResponse(Priority(priority_pred), Category(category_pred), category_confidence)


# Пример использования
if __name__ == "__main__":
    print("Загружаем модель...")
    model = Model("Некоторые параметры")
    response = model.predict("Здравствуйте, заказывал у вас товар месяц назад, всё ещё не доставили")
    print(response.category)
    print(response.priority)
    print(response.confidence)
