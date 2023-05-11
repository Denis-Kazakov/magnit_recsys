import hashlib
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from flask import Flask, render_template, request, redirect, url_for
from .config import *

# Создаем логгер и отправляем информацию о запуске
# Важно: логгер в Flask написан на logging, а не loguru,
# времени не было их подружить, так что тут можно пересоздать 
# logger из logging
logger.add(LOG_FOLDER + "log.log")
logger.info("Наш запуск")

# Создаем сервер и убираем кодирование ответа
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  


@app.route("/<task>")
def main(task: str):
    """
    Эта функция вызывается при вызове любой страницы, 
    для которой нет отдельной реализации

    Пример отдельной реализации: add_data
    
    Параметры:
    ----------
    task: str
        имя вызываемой страницы, для API сделаем это и заданием для сервера
    """
    return render_template('index.html', task=task)

@app.route("/add_data", methods=['POST'])
def upload_file():
    """
    Страница на которую перебросит форма из main 
    Здесь происходит загрузка файла на сервер
    """
    def allowed_file(filename):
        """ Проверяем допустимо ли расширение загружаемого файла """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'add_data'

    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        answer['Сообщение'] = 'Нет файла'
        return answer
    file = request.files['file']

    # Проверяем что путь к файлу не пуст
    if file.filename == '':
        answer['Сообщение'] = 'Файл не выбран'
        return answer
    
    # Загружаем
    if file and allowed_file(file.filename):
        filename = hashlib.md5(file.filename.encode()).hexdigest() 
        file.save(
            os.path.join(
                UPLOAD_FOLDER, 
                filename + file.filename[file.filename.find('.'):]
                )
            )
        answer['Сообщение'] = 'Файл успешно загружен!'
        answer['Успех'] = True
        answer['Путь'] = filename
        return answer
    else:
        answer['Сообщение'] = 'Файл не загружен'
        return answer
        
@app.route("/show_data", methods=['GET'])
def show_file():
    """
    Страница выводящая содержимое файла
    """
   
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'show_file'

    # Проверяем, что указано имя файла
    if 'path' not in request.args:
        answer['Сообщение'] = 'Не указан путь файла'
        return answer
    file = request.args.get('path') 
    
    # Проверяем, что указан тип файла
    if 'type' not in request.args:
        answer['Сообщение'] = 'Не указан тип файла'
        return answer
    type = request.args.get('type')

    file_path = os.path.join(UPLOAD_FOLDER, file + '.' + type)

    # Проверяем, что файл есть
    if not os.path.exists(file_path):
        answer['Сообщение'] = 'Файл не существует'
        return answer

    answer['Сообщение'] = 'Файл успешно загружен!'
    answer['Успех'] = True
    
    # Приводим данные в нужный вид
    if type == 'csv':
        answer['Данные'] = pd.read_csv(file_path).to_dict()
        return answer
    else:
        answer['Данные'] = 'Не поддерживаемы тип'
        return answer


@app.route("/start", methods=['GET'])
def start_model():
    import sys
    import pandas as pd
    from surprise.dump import load


    # Data for the recommendation model
    train = pd.read_csv('my_project/model/train_joke_df.csv')
    test = pd.read_csv('my_project/model/test_joke_df_nofactrating.csv', index_col='InteractionID')
    full = pd.concat((train, test), axis=0, join='outer', ignore_index=True)

    model = load('my_project/model/model_svd')[1]

    def prediction(JID, UID):
        return model.predict(str(int(UID)), str(int(JID))).est


    def ranking(uid, downrate=10):
        '''Joke ratings for a specific user'''
        result = pd.DataFrame({'JID': range(1, 101)})
        # Ratings predicted by the model
        result['predicted_rating'] = result.JID.apply(prediction, UID=uid)

        # Add true ratings
        true_ratings = train.query('UID == @uid').drop(columns='UID')
        result = result.merge(true_ratings, how='left', on='JID')
        missing = result.Rating.isna()
        result.loc[missing, 'Rating'] = result.loc[missing, 'predicted_rating']

        best_joke_index = result.Rating.idxmax()
        best_joke_number = result.JID[best_joke_index]
        best_joke_rating = result.Rating[best_joke_index]

        #Downrate jokes that have already been rated
        rated = full.query('UID == @uid').drop(columns=['UID', 'Rating'])
        rated['downrate'] = downrate
        result = result.merge(rated, how='left', on='JID')
        result.fillna(0, inplace=True)
        result.loc[:, 'Rating'] = result.Rating - result.downrate

        # Select top 10 jokes to recommend
        result.sort_values(by='Rating', ascending=False, inplace=True)
        top10 =  result.JID.iloc[0:10].tolist()
        return [{best_joke_number: best_joke_rating}, top10]


    # path = 'my_project/data/' + sys.argv[1]
    path = 'my_project/data/test.csv'

    df = pd.read_csv(path, index_col=0)
    df['recommendations'] = df.UID.apply(ranking)
    df.to_csv('my_project/data/recommendations.csv')
    
