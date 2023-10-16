from django import forms
from .models import uploadFileModel


class uploadFileForm(forms.ModelForm):

    class Meta:
        model = uploadFileModel
        fields = {'docFile'}

