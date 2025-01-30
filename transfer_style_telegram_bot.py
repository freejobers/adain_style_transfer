from PIL import Image
import torchvision.transforms as transforms
import torch
from adain_nst_model import Model, VGGEncoder, Decoder, RC
import sys

from aiogram import Bot, Dispatcher
from aiogram.types import FSInputFile
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.state import StatesGroup, State
from aiogram import F
import asyncio
import os

# Загрузка изображений
def load_image(image_path, target_size=None):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size) if target_size else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor.squeeze(0)  
    tensor = tensor * std[:, None, None] + mean[:, None, None]  
    tensor = torch.clamp(tensor, 0, 1)  
    return tensor

model_path = 'model.pth'
model = Model()
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("The Telegram bot's token was not found. Make sure that the environment variable TELEGRAM_BOT_TOKEN is set.")

SAVE_DIR = "temp_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Определение состояний
class ImageProcessingState(StatesGroup):
    waiting_for_first_image = State()
    waiting_for_second_image = State()

# Команда /start
@dp.message(Command("start"))
async def start_command(message: Message, state: FSMContext):
    await state.set_state(ImageProcessingState.waiting_for_first_image)
    await message.answer("Привет! Отправьте первое изображение.")

# Прием первого изображения
@dp.message(ImageProcessingState.waiting_for_first_image, F.photo | F.document)
async def first_image(message: Message, state: FSMContext):
    if message.photo:
        # Получаем фото с наивысшим качеством
        photo = message.photo[-1]
        file_id = photo.file_id
        
        file = await bot.get_file(file_id)
        first_image_path = f"{SAVE_DIR}/{file.file_id}"
        
        # Загружаем файл по полученному пути
        await bot.download_file(file.file_path, first_image_path)
    elif message.document and message.document.mime_type.startswith("image/"):
        # Если это изображение как документ
        document = message.document
        file_id = document.file_id
        
        # Получаем файл через bot.get_file
        file = await bot.get_file(file_id)
        first_image_path = f"{SAVE_DIR}/{file.file_id}"
        
        # Загружаем файл по полученному пути
        await bot.download_file(file.file_path, first_image_path)
    else:
        await message.answer("Отправьте изображение в формате фото или файла.")
        return

    # Сохраняем путь к первому изображению в состояние
    await state.update_data(first_image_path=first_image_path)
    await state.set_state(ImageProcessingState.waiting_for_second_image)
    await message.answer("Первое изображение получено! Теперь отправьте второе изображение.")


# Прием второго изображения
@dp.message(ImageProcessingState.waiting_for_second_image, F.photo | F.document)
async def second_image(message: Message, state: FSMContext):
    if message.photo:

        photo = message.photo[-1]
        file_id = photo.file_id
        
        file = await bot.get_file(file_id)
        second_image_path = f"{SAVE_DIR}/{file.file_id}"
        
        # Загружаем файл по полученному пути
        await bot.download_file(file.file_path, second_image_path)
    elif message.document and message.document.mime_type.startswith("image/"):
        # Если это изображение как документ
        document = message.document
        file_id = document.file_id
        
        file = await bot.get_file(file_id)
        second_image_path = f"{SAVE_DIR}/{file.file_id}"
        
        # Загружаем файл по полученному пути
        await bot.download_file(file.file_path, second_image_path)
    else:
        await message.answer("Отправьте изображение в формате фото или файла.")
        return

    # Сохраняем путь ко второму изображению в состояние
    await state.update_data(second_image_path=second_image_path)
    await state.set_state(ImageProcessingState.waiting_for_second_image)
    await message.answer("Второе изображение получено! Ожидайте результат.")

    # Получаем данные о первом изображении из состояния
    data = await state.get_data()
    first_image_path = data.get("first_image_path")

    # Обрабатываем изображения
    generated_image_path = await process_images_with_model(first_image_path, second_image_path)

    # Отправляем результат пользователю
    with open(generated_image_path, "rb") as result_file:
        input_file = FSInputFile(result_file.name)
        await message.answer_photo(input_file, caption="Вот результат обработки!")

    # Удаляем временные файлы
    delete_temp_files(first_image_path, second_image_path, generated_image_path)

    # Завершаем состояние
    await state.clear()

    # Спрашиваем, продолжить ли обработку
    await message.answer("Хотите продолжить обработку?", reply_markup=get_continue_keyboard())


# Клавиатура для вопроса "Продолжить?"
def get_continue_keyboard():
    builder = InlineKeyboardBuilder()
    builder.add(
        InlineKeyboardButton(text="Да", callback_data="continue_yes"),
        InlineKeyboardButton(text="Нет", callback_data="continue_no")
    )
    return builder.as_markup()


# Обработчик ответа "Да" или "Нет"
@dp.callback_query(F.data == "continue_yes")
async def continue_processing(callback_query, state: FSMContext):
    await state.set_state(ImageProcessingState.waiting_for_first_image)
    await callback_query.message.edit_text("Хорошо! Отправьте первое изображение.")

@dp.callback_query(F.data == "continue_no")
async def stop_processing(callback_query, state: FSMContext):
    # Сообщение с инструкцией
    instructions = (
        "Спасибо за использование бота!\n"
        "Чтобы начать обработку заново, отправьте команду /start.\n\n"
        "Инструкция:\n"
        "1. Отправьте два изображения по очереди.\n"
        "2. Получите результат обработки."
    )
    await callback_query.message.edit_text(instructions)

    # Сброс состояния
    await state.clear()    

# Удаление временных файлов
def delete_temp_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

# Функция обработки изображений
async def process_images_with_model(image1_path, image2_path):
    # Загружаем изображения
    content_image = load_image(image1_path, target_size=(512, 512))
    style_image = load_image(image2_path, target_size=(512, 512))

    generate_image = model.generate(content_image, style_image)
    generate_image = denormalize(generate_image).detach().cpu()

    to_pil = transforms.ToPILImage()
    result_image = to_pil(generate_image) 

    # Сохраняем результат
    result_path = f"{SAVE_DIR}/result_image.jpg"
    result_image.save(result_path)
    return result_path


# Основная функция запуска
async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())