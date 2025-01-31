# Adaptive Instance Normalization (AdaIN) Style Transfer

Этот проект реализует перенос стиля изображений с использованием Adaptive Instance Normalization (AdaIN) на PyTorch.

## Описание

Метод AdaIN позволяет смешивать содержимое одного изображения с стилем другого, нормализуя статистики (среднее и стандартное отклонение) активаций слоев нейросети.

Проект включает:

- Код для загрузки и предобработки изображений.
- Реализацию энкодера (VGG-19) и декодера.
- Функцию для вычисления потерь контента и стиля.
- Обучение модели с использованием PyTorch.
- Код для визуализации результатов.

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



Основной код обучения модели находится в блокноте [anain-style-transfer-pytorch.ipynb](https://drive.google.com/file/d/1uPrD1Y_W_zlnF5PrrMae_0dEqMGotcXR/view?usp=sharing). Датасет представляет из себя два набора изображений контентные и стили. Контентые это бычные фотографии, а стили это картины различных художников. Поэтому модель будет хорошо переностить стили на фотографии каких нибудь картин, рисунков.

Модель типа Neural Style Transfer AdaIN для переноса стилей. Обучена на датасете взятый отсюда https://www.kaggle.com/datasets/shaorrran/coco-wikiart-nst-dataset-512-100000.

Модель можно запускать с помощь телеграмм бота, который можно запустить как докер контейнер так и просто запустив [google collaboratory](https://drive.google.com/file/d/1Bl_c6cVGXx1sFJIvf-XjeOG58VeVIImt/view?usp=sharing)

После создания бота в телеграмме и получив токен мы можем запустить наш бот на сервере.

Запуск телеграмм бота из докер контейнера:

1. Клонируем репозиторий
    `git clone https://github.com/freejobers/adain_style_transfer.git`

2. Создаем докер образ 
    `docker build -t telegram-bot .`

3. Запускаем контейнер
    `docker run -e TELEGRAM_BOT_TOKEN=<ваш_токен> telegram-bot`

Для начала работы бота в телеграмме отправляем команду /start и по очереди загружаем два изображения, первое контентное, второе стилевое, и ждем геренерации результата.