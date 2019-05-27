from django import forms
#from uploads.core.models import Document
from django.core.files.uploadedfile import Document

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document', )
