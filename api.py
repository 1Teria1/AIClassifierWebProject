from flask import Flask, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
from TransformerInterface import Model, Category, Priority
import os

app = Flask(__name__)
CORS(app)

print("🔄 Загрузка модели...")
model = Model()
print("✅ Модель загружена!")


def get_category_ru(category):
    mapping = {
        Category.PAYMENT: "Оплата",
        Category.DELIVERY: "Доставка",
        Category.TECH: "Техническая проблема",
        Category.PRODUCT: "Вопрос по товару",
        Category.SPAM: "Спам",
    }
    return mapping.get(category, "Неизвестно")


def get_priority_ru(priority):
    mapping = {
        Priority.HIGH: "Высокий",
        Priority.MEDIUM: "Средний",
        Priority.LOW: "Низкий",
        Priority.SPAM: "Спам",
        Priority.CRITICAL: "Критический",
    }
    return mapping.get(priority, "Неизвестно")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Отсутствует поле text'}), 400
        
        message = data['text'].strip()
        
        if not message:
            return jsonify({'error': 'Пустой текст'}), 400
        
        response = model.predict(message)
        
        result = {
            'category': response.category.name,
            'category_ru': get_category_ru(response.category),
            'priority': response.priority.name,
            'priority_ru': get_priority_ru(response.priority),
            'confidence': response.confidence
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/example-site', methods=['GET'])
def serve_site():
    return send_from_directory('', 'index.html')


@app.route('/', methods=['GET'])
def redirect_to_site():
    return redirect('/example-site')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
