from django.http import JsonResponse
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.camera import VideoCamera
from .models import *
#  MaskDetect, LiveWebCam
# Create your views here.

def NO_of_blink(request):
	search = Contract.objects.filter(user = "hello").last()
	no_blinks = search.no_of_blinks
	dic = {"no_of_blinks": no_blinks}
	return JsonResponse(dic)
def index(request):
	s = Contract.objects.create(user = "hello")
	search = Contract.objects.filter(user = "hello").last()
	no_blinks = search.no_of_blinks
	dic = {"no_of_blinks": no_blinks}

	return render(request, 'streamapp/camera1.html',dic)
def signin(request):
	return render(request, 'streamapp/Signin.html')
def signup(request):
	return render(request, 'streamapp/signup.html')
def Dashboard(request):
	return render(request, 'streamapp/Dashboard.html')
	



def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


def webcam_feed(request):
	return StreamingHttpResponse(gen(IPWebCam()),
					content_type='multipart/x-mixed-replace; boundary=frame')


# def mask_feed(request):
# 	return StreamingHttpResponse(gen(MaskDetect()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')
					
# def livecam_feed(request):
# 	return StreamingHttpResponse(gen(LiveWebCam()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')
