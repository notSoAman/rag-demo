from django.contrib import admin

from chat.models import Chat, Message, User

# registering all models
admin.site.register(Chat)
admin.site.register(Message)
admin.site.register(User)