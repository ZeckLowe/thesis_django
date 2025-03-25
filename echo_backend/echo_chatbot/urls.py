from django.urls import path
from .views import ask_question, store_transcript, transcribe_audio

urlpatterns = [
    path('transcribe/', transcribe_audio, name='transcribe_audio'),
    path('ask/', ask_question, name='ask_question'),
    path('store/', store_transcript, name='store_transcript'),
]