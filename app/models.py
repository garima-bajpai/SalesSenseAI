from __future__ import unicode_literals

from django.db import models

from .validators import validate_file_type


class uploadFileModel(models.Model):

    docFile = models.FileField(verbose_name="File",validators=[validate_file_type])

# Create your models here.


# Create your models here.
