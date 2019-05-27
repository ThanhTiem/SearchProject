from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from first_app import doan
# from .forms import ModelFormWithFileField

# Create your views here.
def index(request):
    my_dict = {'insert_me': "Hello"}
    return render(request,'first_app/index.html', context=my_dict)

def result(request):
    myfile = request.FILES['file']
    fs = FileSystemStorage()
    filename = fs.save(myfile.name, myfile)
    uploaded_file_url = fs.url(filename)
  

    my_dict = doan.exeSearch(uploaded_file_url)
    return render(request,'first_app/search.html', context=my_dict)
