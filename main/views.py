import sqlite3
import time

import pandas as pd
from django.shortcuts import render
from .models import Texts, Modelhist
from .forms import TextForm, UploadFileForm
from django.http import HttpResponse
import os
import pickle
from PyPDF2 import PdfReader
import pymorphy2
from save_model_final import *
from qsstats import QuerySetStats

LOGO_PATH = '/main/logos/%s'


# def index(request):
#    texts = Texts.objects.all()
#    return render(request,'main/index.html',{'title':'Главная страница','texts': texts})


def index(request):  # класс отрисовки главной страницы

    import_classification = False
    result_text = ""  # результат текст
    result_theme = ""  # результат тема
    conn = sqlite3.connect('db.sqlite3')  # подключаемся к бд
    cursor = conn.cursor()
    cursor.execute('''Select * from main_modelhist''')
    re2 = cursor.fetchall()  # берём данные о результатах
    try:
        re2 = re2[-1][1]  # берём последний результат
    except:
        re2 = 0  # если данных о точности модели нет, то она равна 0
    try:
        modeltype = int(cursor.execute("SELECT modeltype FROM main_modelhist ORDER BY id DESC LIMIT 1").fetchone()[0])
    except:
        modeltype = 0
    file = ''
    text_clusification = False  # проверка для вывода поля вывода результата
    right_text = True  # проверка на осмысленный текст
    if request.method == 'POST' and request.POST.get("upload_action"):  # ввод файла
        form = UploadFileForm(request.POST, request.FILES)
        files = request.FILES.getlist('file')  # получаем массив из загруженных файлов
        for i in files:  # цикл прохода по файлам
            try:
                reader = PdfReader(i)  # читаем файл
                text_upl = ""
                for k in range(len(reader.pages)):
                    page = reader.pages[k]
                    text_upl += page.extract_text()
                threshold = 0.2  # обнаружение ввода файла с  неосмылсеным текстом
                osmisl = []
                morph = pymorphy2.MorphAnalyzer()
                for word in text_upl.split():  # сплитим слова и для каждого слова определяем его на бред
                    p = morph.parse(word)
                    score = p[0].score
                    if score >= threshold:  # если совпадение больше 0.2 то добавляем к проверке
                        osmisl.append(word)
                if len(osmisl) >= 4:  # если разумных слов больше 4
                    import_classification = True
                    if modeltype == 1:
                        theme_upl = predict_theme_letter(text_upl, 'main/my_model1.h5', 'main/tokenizer1.pickle',
                                                         'main/encoder.npy')
                    else:
                        theme_upl = predict_theme_letter2(text_upl, 'main/my_model2.h5', 'main/tokenizer2.pickle')
                    # если больше 4 осмысленных слов - запускаем предикт создаём новый класс
                    cursor.execute('''Insert into main_texts(text,theme,checked_by_human,text_accuracy,uploaded,text_file) values(
                    ?,?,?,?,?,?)''', [text_upl, theme_upl, 0, re2, 1, file])
                    conn.commit()
                    result_text = text_upl
                    result_theme = theme_upl  # результат тема

                else:  # иначе текст ложный
                    right_text = False

            except:
                print("Ошибка ввода. Требуется Pdf-файл")

    else:
        form = UploadFileForm()
    if request.method == 'POST' and request.POST.get('_clusification'):  # ввод в текстовую строку
        text_clusification = True  # при прогрузке страницы отобразится поле ответа, нужно для логики прогрузки
        # ответа или ошибки
        text_form = TextForm(request.POST)
        if text_form.is_valid():
            inp_text = text_form['text'].value()  # обнаружение ввода неосмылсеного текста
            threshold = 0.2  # обнаружение ввода текста с  неосмылсеным текстом
            osmisl = []
            morph = pymorphy2.MorphAnalyzer()
            for word in inp_text.split():  # сплитим слова и для каждого слова определяем его на бред
                p = morph.parse(word)
                score = p[0].score
                if score >= threshold:  # если совпадение больше 0.2 то добавляем к проверке
                    osmisl.append(word)
            if len(osmisl) < 4:

                right_text = False
                # если больше 4 осмысленных слов - запускаем предикт создаём новый класс
            else:
                result_text = text_form['text'].value()  # заполняем пустую строку текстом из формы
                if modeltype == 1:
                    # result_theme = predict_theme_letter(result_text, 'main/my_model1.h5', 'main/tokenizer1.pickle',
                    #                                  'main/encoder.npy')
                    result_theme = predict_theme_letter(result_text, 'main/my_model1.h5', 'main/tokenizer1.pickle',
                                                     'main/encoder.npy')
                else:
                    result_theme = predict_theme_letter2(result_text, 'main/my_model2.h5', 'main/tokenizer2.pickle')
                cursor.execute('''Insert into main_texts(text,theme,checked_by_human,text_accuracy,uploaded) values(
                ?,?,?,?,?)''', [result_text, result_theme, 0, re2, 0])

                conn.commit()
        else:
            text_form.save()
    else:
        text_form = TextForm()
    # result = Texts.objects.all().last()  # под результатом подразуемевается последнее значение из бД, на вывод идёт
    # класс, от которого получаем тему в html

    # getting 5 last elements from table
    last_vals = cursor.execute(
        '''Select text,theme,uploaded from main_texts ORDER BY id DESC LIMIT 5''').fetchall()  # берем 5 последних значений

    # прописываем логотипы для отображения в зависимости от типа загрузки

    # проверяем полученные из бд данные на ввод файла и если в поле uploaded число 1, то переопределяем каждый логотип на file_logo
    logo1 = LOGO_PATH % 'text_logo.png'  # прописываем логотипы для отображения в зависимости от типа загрузки

    logo2 = LOGO_PATH % 'text_logo.png'

    logo3 = LOGO_PATH % 'text_logo.png'

    logo4 = LOGO_PATH % 'text_logo.png'

    logo5 = LOGO_PATH % 'text_logo.png'

    try:
        if int(last_vals[0][2]):
            logo1 = LOGO_PATH % 'file_logo.png'
    except:
        logo1 = LOGO_PATH % 'text_logo.png'
    try:
        if int(last_vals[1][2]):
            logo2 = LOGO_PATH % 'file_logo.png'
    except:
        logo2 = LOGO_PATH % 'text_logo.png'
    try:
        if int(last_vals[2][2]):
            logo3 = LOGO_PATH % 'file_logo.png'
    except:
        logo3 = LOGO_PATH % 'text_logo.png'
    try:
        if int(last_vals[3][2]):
            logo4 = LOGO_PATH % 'file_logo.png'
    except:
        logo4 = LOGO_PATH % 'text_logo.png'
    try:
        if int(last_vals[4][2]):
            logo5 = LOGO_PATH % 'file_logo.png'
    except:
        logo5 = LOGO_PATH % 'text_logo.png'
    while len(last_vals) < 5:
        last_vals.append(["пример", "тема примера"])

    try:
        res_mass = [0, 0, 0, 0, 0]
        results = cursor.execute('''Select text_accuracy from main_texts ORDER BY id DESC LIMIT 5''').fetchall()
        for i in range(len(results)):
            res_mass[i] = results[i]
        res1 = float("{0:.3f}".format(res_mass[0][0]))
        res2 = float("{0:.3f}".format(res_mass[0][0]))
        res3 = float("{0:.3f}".format(res_mass[0][0]))
        res4 = float("{0:.3f}".format(res_mass[0][0]))
        res5 = float("{0:.3f}".format(res_mass[0][0]))
    except:
        res_mass = [[0.0], [0.0], [0.0], [0.0], [0.0]]
        res1 = float("{0:.3f}".format(res_mass[0][0]))
        res2 = float("{0:.3f}".format(res_mass[0][0]))
        res3 = float("{0:.3f}".format(res_mass[0][0]))
        res4 = float("{0:.3f}".format(res_mass[0][0]))
        res5 = float("{0:.3f}".format(res_mass[0][0]))
    return render(request, 'main/index.html',
                  {'text_form': text_form,
                   'text_clusification': text_clusification,
                   'title': 'Главная страница',
                   'result_text': result_text,
                   'result_theme': result_theme,
                   'accuracy': float("{0:.3f}".format(re2)),
                   'form': form,
                   'file': str(file),
                   'text1': last_vals[0][0],
                   'text2': last_vals[1][0],
                   'text3': last_vals[2][0],
                   'text4': last_vals[3][0],
                   'text5': last_vals[4][0],
                   'theme1': last_vals[0][1],
                   'theme2': last_vals[1][1],
                   'theme3': last_vals[2][1],
                   'theme4': last_vals[3][1],
                   'theme5': last_vals[4][1],
                   'logo1': logo1,
                   'logo2': logo2,
                   'logo3': logo3,
                   'logo4': logo4,
                   'logo5': logo5,
                   'right_text': right_text,
                   'import_classification': import_classification,
                   'res1': res1,
                   'res2': res2,
                   'res3': res3,
                   'res4': res4,
                   'res5': res5,

                   })


# def index(request):
#     if request.method == 'POST':
#         file = request.FILES['file'].read()
#         fileName = request.POST['filename']
#         existingPath = request.POST['existingPath']
#         end = request.POST['end']
#         nextSlice = request.POST['nextSlice']
#
#         if file == "" or fileName == "" or existingPath == "" or end == "" or nextSlice == "":
#             res = JsonResponse({'data': 'Invalid Request'})
#             return res
#         else:
#             if existingPath == 'null':
#                 path = 'media/' + fileName
#                 with open(path, 'wb+') as destination:
#                     destination.write(file)
#                 FileFolder = Texts()
#                 FileFolder.existingPath = fileName
#                 FileFolder.eof = end
#                 FileFolder.name = fileName
#                 FileFolder.save()
#                 if int(end):
#                     res = JsonResponse({'data': 'Uploaded Successfully', 'existingPath': fileName})
#                 else:
#                     res = JsonResponse({'existingPath': fileName})
#                 return res
#
#             else:
#                 path = '/main/media/' + existingPath
#                 model_id = Texts.objects.get(existingPath=existingPath)
#                 if model_id.name == fileName:
#                     if not model_id.eof:
#                         with open(path, 'ab+') as destination:
#                             destination.write(file)
#                         if int(end):
#                             model_id.eof = int(end)
#                             model_id.save()
#                             res = JsonResponse({'data': 'Uploaded Successfully', 'existingPath': model_id.existingPath})
#                         else:
#                             res = JsonResponse({'existingPath': model_id.existingPath})
#                         return res
#                     else:
#                         res = JsonResponse({'data': 'EOF found. Invalid request'})
#                         return res
#                 else:
#                     res = JsonResponse({'data': 'No such file exists in the existingPath'})
#                     return res
#     return render(request, 'main/index.html')

def learning(request):  # класс отрисовки страницы переобучения
    if request.POST.get('run_script'):
        relearn_model()  # переобучение модеои
        if request.method == 'POST' and request.POST.get("_getresult"):
            return model(request)  # перенаправляем на страницу с результатами
        return render(request, 'main/learning.html', {'learn': True})  # при learn получаем вывод об успешном обучении
    if request.POST.get('run_script1'):
        relearn_model2()  # переобучение модеои
        if request.method == 'POST' and request.POST.get("_getresult"):
            return model(request)  # перенаправляем на страницу с результатами
        return render(request, 'main/learning.html', {'learn': True})  # при learn получаем вывод об успешном обучении
    if request.method == 'POST' and request.POST.get("_getresult"):
        return model(request)  # перенаправляем на страницу с результатами
    return render(request, 'main/learning.html', {'learn': False})  # иначе без прогресс бара


def model(request):  # класс отрисовки страницы результатов
    conn = sqlite3.connect('db.sqlite3')  # подключаемся к бд
    cursor = conn.cursor()
    cursor.execute('''Select * from main_modelhist''')  # берём результаты модели из бд
    result = cursor.fetchall()

    try:
        re1 = result[-2][1]  # предпоследний результат
        re2 = result[-1][1]  # последний результат
    except:
        if len(result) == 1:
            re1 = 0
            re2 = result[-1][1]
        else:
            re1 = 0
            re2 = 0
    diff = re2 - re1
    if diff < 0:  # проверка на разницу
        re3 = f'Точность снизилась на {float("{0:.3f}".format(abs(diff)))}'
    elif diff > 0:
        re3 = f'Точность повысилась на {float("{0:.3f}".format(abs(diff)))}'
    else:
        re3 = 'Точность не изменилась'
    re1 = float("{0:.3f}".format(re1))  # форматируем до 3х цифр после запятой
    re2 = float("{0:.3f}".format(re2))

    item = Modelhist.objects.all().values()  # готовим данные для графика
    df = pd.DataFrame(item)
    df1 = df.date.tolist()
    for i in range(len(df1)):
        df1[i] = str(df1[i])  # делаем дату строкой
    df = df['result'].tolist()  # преобразуем в массив numpy
    return render(request, 'main/model.html', {'result1': re1, 'result2': re2, 'result3': re3, 'df': df, 'df1': df1})


def about(request):  # страница о проекте
    return render(request, 'main/about.html')
# def ajax_file_upload(request):
#     return render(request,"ajax_file_upload.html")
# def ajax_file_upload_save(request):
#     print(request.POST)
#     print(request.FILES)
