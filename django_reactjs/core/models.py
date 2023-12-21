from django.db import models

# Create your models here.
class Files(models.Model):
    img = models.FileField(upload_to='store/image')
    def __str__(seft):
        return seft.img