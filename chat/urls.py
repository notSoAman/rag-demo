from django.urls import path
from . import views



urlpatterns = [
    path("", views.index, name="index"),
    path("ask/", views.ask, name="ask"),
    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("chat/<int:chat_id>/", views.chat, name="chat"),
]