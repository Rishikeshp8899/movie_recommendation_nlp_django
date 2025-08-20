from django.db import models
from django.core.exceptions import ValidationError

class Prediction(models.Model):
    query = models.CharField(max_length=100)
    prediction = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if self.pk:  # Prevent updates
            raise ValidationError("Updates are not allowed for this model.")
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        raise ValidationError("Deletion is not allowed for this model.")

    def __str__(self):
        return self.query
