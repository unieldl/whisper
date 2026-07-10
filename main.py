import whisper
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

# создание нужных папок
def check_and_create_folders(current_dir):
    REQUIRED_FOLDERS = ['models', 'input', 'results']

    for folder_name in REQUIRED_FOLDERS:
        folder_path = current_dir / folder_name
        folder_path.mkdir(exist_ok=True)


# проверка есть модель в папке или нет
def check_model(model_name, current_dir):
    folder_path = current_dir / 'models'
    file_path = folder_path / f'{model_name}.pt'
    return file_path.is_file()


# проход по всем моделям
# если нет в папке - скачивание
def load_models(current_dir):
    models = ['tiny', 'base']
        # 'small': '244 M',
        # 'medium': '769 M',
        # 'large': '1550 M',
        # 'turbo': '809 M'

    model_directory = current_dir / 'models'
    for model_name in models:
        if not check_model(model_name, current_dir): 
            whisper.load_model(model_name, download_root=str(model_directory))
    

# выбор модели
def model_choice(current_dir):
    folder_path = current_dir / 'models'
    all_models_list = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

    for index, model_name in enumerate(all_models_list):
        print(f"{index+1}. {model_name}")
    
    model_index = int(input('Выбери модель по номеру в списке (рекомендуемая - small)\n'))
    model_path = './models/' + all_models_list[model_index - 1]
    return model_path

# выбор конкретного файла из директории
def file_choice():
    all_audio_path = './input/'
    all_audio_list = os.listdir(all_audio_path)

    print('Список файлов в папке "input" для транскрипции')
    for file_index in range(len(all_audio_list)):
        file_num = file_index + 1
        print(file_num, all_audio_list[file_index])

    audio_choice = int(input('Введи номер файла в списке\n'))
    audio_path = './input/' + all_audio_list[audio_choice - 1]
    audio_name = all_audio_list[audio_choice - 1]
    return audio_name, audio_path

# старт транскрипции
def model_start(model_path, audio_path):
    print('Старт транскрипции..')
    model = whisper.load_model(model_path)
    result = model.transcribe(audio_path, fp16=False, verbose=False, temperature=0.0, condition_on_previous_text=False, beam_size=1)
    return result

# создание тхт файла в папке резалтс + запись результата в тхт файл
def create_file(audio_name, result):
    index = audio_name.rfind('.')
    file_name = audio_name[:index]

    file_txt = open('./results/' + file_name + '.txt', 'w+', encoding='utf-8')
    file_txt.write(result['text'])
    print('Готово! Транскрипция файла в папке "results".')

def transcription():
    check_and_create_folders(CURRENT_DIR)
    load_models(CURRENT_DIR)

    try:
        model_path = model_choice(CURRENT_DIR)
    except ValueError:
        print('Неправильный ввод. Попробуйте еще раз')
        model_path = model_choice(CURRENT_DIR)

    try:
        audio_name, audio_path = file_choice()
    except ValueError:
        print('Неправильный ввод. Попробуйте еще раз')
        audio_name, audio_path = file_choice()

    result = model_start(model_path, audio_path)
    create_file(audio_name, result)

def main():
    try:
        transcription()  
    except KeyboardInterrupt:
        print('Завершение работы..')
        return
    except Exception as e:
        print(f'Произошла ошибка -> {e}\nПроверьте наличие программы ffmpeg на компьютере\nУстановите по ссылке -> https://www.ffmpeg.org/download.html\n')

if __name__ == '__main__':
    main()