from django.urls import path
from .views import transcribe_audio, ask_question, store_transcript, store_single_speaker, get_pinecone_stats, test_voice_similarity, send_meeting_email

urlpatterns = [
    path('transcribe/', transcribe_audio, name='transcribe_audio'),
    path('ask/', ask_question, name='ask_question'), 
    path('store/', store_transcript, name='store_transcript'), 
    path('email/', send_meeting_email, name='send_meeting_email'),
    path('store-speaker/', store_single_speaker, name='store_single_speaker'),
    path('pinecone-stats/', get_pinecone_stats, name='pinecone_stats'),
    path('test-similarity/', test_voice_similarity, name='test_voice_similarity'),
]