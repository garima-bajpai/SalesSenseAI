from __future__ import unicode_literals
from .validators import validate_file_type

from django.db import models


class uploadFileModel(models.Model):

    docFile = models.FileField(verbose_name="File",validators=[validate_file_type])
