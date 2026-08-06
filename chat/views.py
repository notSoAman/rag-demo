from django.shortcuts import render
from chat.services.llm import generate_answer

# Create your views here.
def index(request):
    return render(request, "index.html")


def ask(request):

    question = request.POST.get("prompt")
    answer = generate_answer(question)

    return render(
        request,
        "partials/answer.html",
        {
            "question": question,
            "answer": answer
        }
    )