from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, "index.html")


def ask(request):

    question = request.POST.get("prompt")

    return render(
        request,
        "partials/answer.html",
        {
            "question": question,
            "answer": "This is a fake answer."
        }
    )