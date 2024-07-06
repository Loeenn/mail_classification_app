# Приложение автоматизированной классификации почтовых сообщений
Инструкция по установке\
1.Скопируте ссылку на репозиторий и в терминале введите git clone *ссылка на проект* в папке, где хотите запустить проект.\
2. Установите Python 3.10.5\
3. Откроте терминал питона и Создание вирутального окружения\
python -m venv env *адрес папки где делали git clone*\
4.Scripts\activate \
(Если не работает открываем powershell от имени идминистратора и прописываем \
Set-ExecutionPolicy RemoteSigned\
A\
5.Зайдите в папку проекта и пропишите команду ниже:
pip install -r requirements.txt
6.cd django_project_main/taskmanager\
7. python manage.py makemigrations\
8. python manage.py migrate\
9. python  manage.py createsuperuser и создаете логин и пароль для вашей базы данных\
10.python python manage.py runserver\
11.При любых проблемах с опредлением темы, стоит переобучить модель. Модули кераса могут иногда сдавать сбои.
