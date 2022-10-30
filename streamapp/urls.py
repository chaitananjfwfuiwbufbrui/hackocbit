from django.urls import path, include
from streamapp import views

from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    # path('webcam_feed', views.webcam_feed, name='webcam_feed'),
    # path('mask_feed', views.mask_feed, name='mask_feed'),
	# path('livecam_feed', views.livecam_feed, name='livecam_feed'),
    ] + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
