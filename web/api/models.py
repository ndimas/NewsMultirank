from django.db import models

class Prediction(models.Model):
    result = models.CharField(max_length=100)

    def __str__(self):
        return self.name