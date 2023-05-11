﻿# 2-й этап соревнования по рекомендательным системам ВШЭ - Магнит



### Комментарии по запуску:

1. Файл с данными для проверки алгоритма должен лежать в папке my_project/data/ и называться test.csv. Итоговые рекомендации будут там же в файле с названием recommendations.csv. В этой версии пока не получилось загружать внешний файл. В браузере вижу сообщение об ошибке, но мой код работает и выдает рекомендации.
2. Файл с моделью заархивирован (model_svd.zip), т.к. GitHub не принимает большие файлы. Перед запуском необходимо разархивировать.



В идеале лучше предлагать пользователю только новые анекдоты (те, что он пока не оценивал). Но в данном случае матрица оценок не разреженная (заполнение 58%, с учетом тестового набора 72%). Каждый пользователь оценил минимум 36 анекдотов. Макс. число анекдотов, оцененных пользователем, в обучающей выборке - 95, с учетом тестовой - 100 (общее число анекдотов), поэтому приписываю уже оценненым анекдотам более низкий вес (снижаю оценку), но не удаляю совсем.



Для предсказания рейтингов использовал ту же модель, что и на первом этапе соревнования - matrix factorization (был перегружен текущей работой и не успел использовать более интересные варианты).



Создание образа:



```bash
docker build --pull --rm -f "Dockerfile" -t my-project "."
```



Запуск контейнера:



``` bash
docker run --rm -p 5000:5000 -v $(pwd)/my_project/log:/my_project/log -v $(pwd)/my_project/data:/my_project/data --name my_script my-project:latest
```


