from django.shortcuts import render
from chat.services.llm import generate_answer
from chat.services.markdown import markdown_to_html

# Create your views here.
def index(request):
    return render(request, "index.html")


def ask(request):

    question = request.POST.get("prompt")
    answer = generate_answer(question)
    answer_html = markdown_to_html(answer)

    return render(
        request,
        "partials/answer.html",
        {
            "question": question,
            "answer": answer_html
        }
    )