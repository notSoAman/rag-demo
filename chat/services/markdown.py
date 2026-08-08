from markdown_it import MarkdownIt


_markdown_renderer = MarkdownIt("commonmark", {"html": False})


def markdown_to_html(markdown_text: str) -> str:
    if not markdown_text:
        return ""

    return _markdown_renderer.render(markdown_text)