from django.test import TestCase
from django.contrib.auth import get_user_model
from django.urls import reverse
from .models import Chat, Message


class AskViewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="tester",
            password="secret123"
        )

    def test_first_question_creates_chat_and_pushes_new_url(self):
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("ask"),
            {"prompt": "Who is Zeus?"},
            HTTP_HX_REQUEST="true"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Chat.objects.count(), 1)
        self.assertEqual(Message.objects.count(), 1)
        self.assertEqual(response["HX-Push-Url"], "/chat/1/")

    def test_subsequent_question_uses_existing_chat_without_changing_url(self):
        self.client.force_login(self.user)
        chat = Chat.objects.create(user=self.user, title="Test chat")
        Message.objects.create(chat=chat, question="Hello", answer="Hi")

        response = self.client.post(
            reverse("ask"),
            {"prompt": "What about Hera?", "chat_id": chat.id},
            HTTP_HX_REQUEST="true"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Chat.objects.count(), 1)
        self.assertEqual(Message.objects.count(), 2)
        self.assertNotIn("HX-Push-Url", response)
