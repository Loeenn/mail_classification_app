import pymorphy2
from PyPDF2 import PdfReader
from django.contrib import admin

from django import forms
from .forms import UploadFileForm
from .models import Texts, Modelhist
from django_object_actions import DjangoObjectActions
from django.contrib import admin
from django.http import HttpResponseRedirect
from django.urls import re_path
import sqlite3
import pandas as pd
from save_model_final import predict_theme_letter
from django.shortcuts import render
from .forms import TextForm, UploadFileForm


@admin.register(Texts)
class PersonAdmin(admin.ModelAdmin):
    change_list_template = "admin/model_change_list.html"  # дополняем админку html кодом

    def get_urls(self):
        urls = super(PersonAdmin, self).get_urls()  # получаем сслыку на админку
        custom_urls = [
            re_path('^import1/$', self.insert1, name='insert1'),  # дописываем пути к функциям
            re_path('^import2/$', self.insert2, name='insert2'),
        ]
        return custom_urls + urls  # возвращаем редиректы для прописанных кнопок в model_change_list

    def insert1(self, request):
        count = 0
        conn = sqlite3.connect('db.sqlite3')  # подключение к бд
        cursor = conn.cursor()
        cursor.execute('''Delete from main_texts''')  # очищаем бд
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                files = request.FILES.getlist('file')
                for i in files:
                    df = pd.read_excel(i)
                    count += df.shape[0]
                    if len(df.iloc[0]) >= 2:
                        # column_names = list(df.columns.values)
                        # themes = df[column_names[0]]
                        # texts = df[column_names[1]]
                        themes = df['Тематика письма']
                        texts = df['Суть обращения']
                        for j in range(len(themes)):
                            try:
                                cursor.execute('''
                                         INSERT INTO main_texts(text,checked_by_human,theme,text_accuracy,uploaded) Values (?,?,?,1,0)

                                         ''', [str(texts[j]), 0, str(themes[j])])
                                conn.commit()
                            except:
                                pass
                        # убираем дубликаты
                        cursor.execute('''DELETE FROM main_texts WHERE id NOT IN (
                                                 SELECT MAX(id) 
                                                 FROM main_texts
                                                 GROUP BY text)''')
                        conn.commit()
                        count -= cursor.execute('''select count(*)FROM main_texts WHERE id NOT IN (
                                         SELECT MAX(id) 
                                         FROM main_texts
                                         GROUP BY text)''').fetchone()[0]
                        self.message_user(request, f"создано {count} новых записей")
                        return HttpResponseRedirect("../")
                    else:
                        column_names = list(df.columns.values)
                        # texts = df[column_names[0]]
                        texts = df['Суть обращения']
                        cursor.execute('''Select * from main_modelhist''')
                        re2 = cursor.fetchall()  # берём данные о результатах
                        try:
                            re2 = re2[-1][1]  # берём последний результат
                        except:
                            re2 = 0  # если данных о точности модели нет, то она равна 0
                        for j in range(len(texts)):
                            try:
                                cursor.execute('''
                                         INSERT INTO main_texts(text,checked_by_human,theme,text_accuracy,uploaded) Values (?,?,?,?,0)

                                         ''', [str(texts[j]), 0,
                                               str(predict_theme_letter(texts, 'my_model.h5', 'tokenizer.pickle',
                                                                        'encoder.npy')), re2])
                            except:
                                pass
                            conn.commit()
                        # убираем дубликаты
                        cursor.execute('''DELETE FROM main_texts WHERE id NOT IN (
                                                 SELECT MAX(id) 
                                                 FROM main_texts
                                                 GROUP BY text)''')
                        count -= cursor.execute('''select count(*)FROM main_texts WHERE id NOT IN (
                                         SELECT MAX(id) 
                                         FROM main_texts
                                         GROUP BY text)''').fetchone()[0]
                        conn.commit()

        self.message_user(request, f"создано {count} новых записей")
        return HttpResponseRedirect("../")

    def changelist_view(self, *args, **kwargs):
        try:
            view = super().changelist_view(*args, **kwargs)
            view.context_data['submit_csv_form'] = UploadFileForm
        except:
            return HttpResponseRedirect("../")
        return view

    def insert2(self, request):
        count = 0
        conn = sqlite3.connect('db.sqlite3')
        cursor = conn.cursor()
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                files = request.FILES.getlist('file')
                for i in files:
                    df = pd.read_excel(i)
                    count += df.shape[0]
                    if len(df.iloc[0]) >= 2:
                        # column_names = list(df.columns.values)
                        # themes = df[column_names[0]]
                        # texts = df[column_names[1]]
                        themes = df['Тематика письма']
                        texts = df['Суть обращения']
                        for j in range(len(themes)):
                            try:
                                cursor.execute('''
                                 INSERT INTO main_texts(text,checked_by_human,theme,text_accuracy,uploaded) Values (?,?,?,1,0)

                                 ''', [str(texts[j]), 0, str(themes[j])])
                                conn.commit()
                            except:
                                pass
                        # убираем дубликаты
                        cursor.execute('''DELETE FROM main_texts WHERE id NOT IN (
                                         SELECT MAX(id)
                                         FROM main_texts
                                         GROUP BY text)''')
                        count -= cursor.execute('''select count(*) FROM main_texts WHERE id NOT IN (
                                         SELECT MAX(id) 
                                         FROM main_texts
                                         GROUP BY text)''').fetchone()[0]
                        conn.commit()
                        self.message_user(request, f"создано {count} новых записей")
                        return HttpResponseRedirect("../")
                    else:
                        # column_names = list(df.columns.values)
                        # texts = df[column_names[0]]
                        texts = df['Суть обращения']
                        cursor.execute('''Select * from main_modelhist''')
                        re2 = cursor.fetchall()  # берём данные о результатах
                        try:
                            re2 = re2[-1][1]  # берём последний результат
                        except:
                            re2 = 0  # если данных о точности модели нет, то она равна 0
                        for j in range(len(texts)):
                            try:
                                cursor.execute('''
                                 INSERT INTO main_texts(text,checked_by_human,theme,text_accuracy,uploaded) Values (?,?,?,?,0)

                                 ''', [str(texts[j]), 0,
                                       str(predict_theme_letter(texts, 'my_model.h5', 'tokenizer.pickle',
                                                                'encoder.npy')), re2])
                            except:
                                pass
                            conn.commit()
                        # убираем дубликаты
                        cursor.execute('''DELETE FROM main_texts WHERE id NOT IN (
                                         SELECT MAX(id)
                                         FROM main_texts
                                         GROUP BY text)''')
                        count -= cursor.execute('''select count(*) FROM main_texts WHERE id NOT IN (
                                         SELECT MAX(id) 
                                         FROM main_texts
                                         GROUP BY text)''').fetchone()[0]
                        conn.commit()
        self.message_user(request, f"создано {count} новых записей")
        return HttpResponseRedirect("../")

    list_display = ["text", "theme", "checked_by_human"]
    list_editable = ["theme", "checked_by_human"]


@admin.register(Modelhist)
class BigAdmin(admin.ModelAdmin):
    list_display = ["date", "result"]
