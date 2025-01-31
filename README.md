# Adaptive Instance Normalization (AdaIN) Style Transfer

Этот проект реализует перенос стиля изображений с использованием Adaptive Instance Normalization (AdaIN) на PyTorch.

## Описание

Метод AdaIN позволяет смешивать содержимое одного изображения с стилем другого, нормализуя статистики (среднее и стандартное отклонение) активаций слоев нейросети.

Основной код обучения модели находится в блокноте `adain-style-transfer-pytorch.ipynb`. Датасет состоит из двух наборов изображений: контентных и стилевых. Контентные изображения — это обычные фотографии, а стилевые — картины различных художников. Поэтому модель хорошо переносит стили картин и рисунков на фотографии.

Проект включает:
- Код для загрузки и предобработки изображений.
- Реализацию энкодера (VGG-19) и декодера.
- Функцию для вычисления потерь контента и стиля.
- Обучение модели с использованием PyTorch.
- Код для визуализации результатов.

Модель типа Neural Style Transfer (AdaIN) обучена на датасете, взятом с [Kaggle](https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000).

## Установка

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

## Использование

1. **Подготовьте данные:**
   - Папка с изображениями контента.
   - Папка с изображениями стиля.

2. **Запустите Jupyter Notebook:**
   ```bash
   jupyter notebook adain-style-transfer-pytorch.ipynb
   ```

3. **Выполните ячейки по порядку.**
   - Код загрузит модель VGG-19, применит AdaIN и обучит декодер для генерации изображений.

4. **Сохранение результата:**
   ```python
   from torchvision.utils import save_image
   save_image(output, "stylized_output.jpg")
   ```

## Телеграм-бот

Модель можно запустить с помощью телеграм-бота, который поддерживает запуск как в Docker-контейнере, так и в Google Colab.

### Запуск телеграм-бота в Docker

1. **Клонируем репозиторий:**
   ```bash
   git clone https://github.com/freejobers/adain_style_transfer.git
   ```
2. **Создаем Docker-образ:**
   ```bash
   docker build -t telegram-bot .
   ```
3. **Запускаем контейнер:**
   ```bash
   docker run -e TELEGRAM_BOT_TOKEN=<ваш_токен> telegram-bot
   ```

После создания бота в Telegram и получения токена можно запустить его на сервере.

Для начала работы в Telegram отправьте команду `/start`, затем загрузите два изображения: 
- Первое — контентное (фотография).
- Второе — стилевое (картина).

Дождитесь генерации результата.

## Пример кода

```python
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

model = Model()
content_img = Image.open("path_to_content.jpg")
style_img = Image.open("path_to_style.jpg")

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

content_tensor = transform(content_img).unsqueeze(0)
style_tensor = transform(style_img).unsqueeze(0)

output = model.generate(content_tensor, style_tensor, alpha=0.8)
plt.imshow(output.squeeze().permute(1, 2, 0))
plt.show()
```

Этот проект основан на идее работы "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization".

