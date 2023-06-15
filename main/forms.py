from .models import Texts
from django.forms import ModelForm, TextInput


class TextForm(ModelForm):
    class Meta:
        model = Texts
        fields = ('text',)
        # fields = ["text","theme"]
        widgets = {
            "text": TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Введите текст',
                })
        }
        #    "theme": TextInput(attrs={
        #        'class': 'form-control',
        #        'placeholder': 'Введите тему'
        #   }),
        # }
from django import forms
from django.forms import ClearableFileInput
class UploadFileForm(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True})) #класс для ввода нескольких файлов