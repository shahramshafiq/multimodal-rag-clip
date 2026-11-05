"""Step 6: send the question, the text chunks and the figure pictures to a vision model and get the answer."""
import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI, OpenAIError

log = logging.getLogger("answer")

INSTRUCTIONS = """You answer questions using only the context you are given: text passages and figures from documents.
- Look at the figures carefully. Read chart values, legends, colors and axes from the picture when the answer is there.
- Use only what the context shows. Do not use outside knowledge.
- End the answer with its source in brackets, for example [noaa_sea_ice_2020, page 4] or [noaa_sea_ice_2020, Figure 2].
- Keep the answer short.
- If the context does not contain the answer, reply with exactly this sentence and nothing else: {not_found}"""


@dataclass
class Answer:
    text: str
    input_tokens: int
    output_tokens: int


def _image_as_data_url(path):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_content(question, retrieved, detail):
    """The user message: the question, then the pages, then each figure with its caption."""
    content = [{"type": "input_text", "text": f"Question: {question}\n\nText passages:"}]
    for page in retrieved.pages:
        content.append({"type": "input_text", "text": f"[{page['doc']}, page {page['page']}] {page['text']}"})
    for figure in retrieved.figures:
        label = f"[{figure['doc']}, Figure {figure['number']}, page {figure['page']}] {figure['caption']}"
        content.append({"type": "input_text", "text": label})
        content.append({"type": "input_image", "image_url": _image_as_data_url(figure["image_path"]), "detail": detail})
    return content


class Answerer:
    def __init__(self, settings):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty. Add it to .env in the project folder.")
        self.settings = settings
        self.client = OpenAI(api_key=api_key, timeout=settings.openai_timeout_seconds,
                             max_retries=settings.openai_max_retries)

    def answer(self, question, retrieved):
        s = self.settings
        try:
            response = self.client.responses.create(
                model=s.openai_model,
                instructions=INSTRUCTIONS.format(not_found=s.not_found_sentence),
                input=[{"role": "user", "content": build_content(question, retrieved, s.image_detail)}],
                max_output_tokens=s.answer_max_tokens,
            )
        except OpenAIError as exc:
            log.error("the model call failed: %s", type(exc).__name__)
            raise RuntimeError(f"The answer could not be generated: {exc}") from exc
        return Answer(response.output_text.strip(), response.usage.input_tokens, response.usage.output_tokens)
