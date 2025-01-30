# Проект нейронной сети NST AdaIN для переноса стилей изображения

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