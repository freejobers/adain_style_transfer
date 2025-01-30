# Используем официальный Python-образ (slim-версия для меньшего размера)
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY model.pth adain_nst_model.py transfer_style_telegram_bot.py requirements.txt ./

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Создаем директорию для временных файлов
RUN mkdir -p /app/temp_images

# Устанавливаем переменные окружения
ENV SAVE_DIR=/app/temp_images

# Добавляем пользователя и назначаем права
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

# Запуск бота
ENTRYPOINT ["python", "-u", "transfer_style_telegram_bot.py"]