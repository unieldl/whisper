import whisper
import os

# выбор модели
def model_choice():
    all_models_path = './models/'
    all_models_list = os.listdir(all_models_path)

    for model_index in range(len(all_models_list)):
        model_num = model_index + 1
        print(model_num, all_models_list[model_index])
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

    try:
        model_path = model_choice()
    except ValueError:
        print('Неправильный ввод. Попробуйте еще раз')
        model_path = model_choice()

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

if __name__ == '__main__':
    main()