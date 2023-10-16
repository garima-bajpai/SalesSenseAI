
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.exceptions import ValidationError

def validate_file_type(upload):
    valid_contents=['text/csv','application/vnd.ms-excel']
    max_size="20971520"
    if upload.file.size > max_size:
        raise ValidationError('File size limit exceeded')
    if upload.file.content_type not in valid_contents:
        raise ValidationError('File type invalid')