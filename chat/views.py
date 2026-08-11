from django.http import HttpResponse, request
from django.shortcuts import render, redirect, get_object_or_404
from chat.services.llm import generate_answer
from chat.services.markdown import markdown_to_html
from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate, login, logout
from .models import Chat, Message
from django.template.loader import render_to_string
User = get_user_model()

# Create your views here.
def index(request):
    chats = []

    if request.user.is_authenticated:
        chats = Chat.objects.filter(
            user=request.user
        ).order_by("-updated_at")

    return render(
        request,
        "index.html",
        {
            "chats": chats
        }
    )

def ask(request, chat_id=None):
    question = request.POST.get("prompt", "").strip()

    if not question:
        return HttpResponse("Please enter a question.", status=400)

    chat = None
    new_chat = False

    incoming_chat_id = (
        request.POST.get("chat_id")
        or request.GET.get("chat_id")
        or chat_id
    )

    # Logged-in users
    if request.user.is_authenticated:

        if incoming_chat_id:
            chat = get_object_or_404(
                Chat,
                id=int(incoming_chat_id),
                user=request.user
            )

        else:
            chat = Chat.objects.create(
                user=request.user,
                title=question[:50]
            )
            new_chat = True

    # Generate answer
    answer = generate_answer(question)

    # Save message
    if chat is not None:
        Message.objects.create(
            chat=chat,
            question=question,
            answer=answer
        )

    # FIRST QUESTION OF A NEW CHAT
    if new_chat:
        response = HttpResponse()
        response["HX-Redirect"] = f"/chat/{chat.id}/"
        return response

    # Existing chat / anonymous user
    answer_html = markdown_to_html(answer)

    html = render_to_string(
        "partials/answer.html",
        {
            "question": question,
            "answer": answer_html,
        },
        request=request
    )

    return HttpResponse(html)
    
    
    
def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]

        if password != confirmation:
            return render(
                request,
                "register.html",
                {"message": "Passwords must match."}
            )

        if User.objects.filter(username=username).exists():
            return render(
                request,
                "register.html",
                {"message": "Username already exists."}
            )

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        login(request, user)

        return redirect("index")

    return render(request, "register.html")



def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is None:
            return render(
                request,
                "login.html",
                {"message": "Invalid username or password."}
            )

        login(request, user)

        return redirect("index")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("index")

def chat(request, chat_id):
    if not request.user.is_authenticated:
        return redirect("login")

    chat = get_object_or_404(
        Chat,
        id=chat_id,
        user=request.user
    )

    messages = Message.objects.filter(
        chat=chat
    ).order_by("created_at")

    for message in messages:
        message.answer_html = markdown_to_html(
            message.answer
        )

    chats = Chat.objects.filter(
        user=request.user
    ).order_by("-updated_at")

    return render(
        request,
        "chat.html",
        {
            "chat": chat,
            "messages": messages,
            "chats": chats,
        }
    )