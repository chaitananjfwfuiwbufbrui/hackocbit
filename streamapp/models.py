from email.policy import default
from pyexpat import model
from django.db import models
from video_stream import settings

# Create your models here.
class Contract(models.Model):
    user = models.CharField(max_length=30,default = "chay",)
    no_of_blinks = models.IntegerField(default = 0)
    