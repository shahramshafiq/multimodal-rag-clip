"""Step 6: send the question, the pages and the figure pictures to a vision model and get the answer."""
import base64
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI, OpenAIError

log = logging.getLogger("answer")

INSTRUCTIONS = """You answer questions using only the context you are given: text passages and figures from documents.

Rules:
- Use only what the context shows. Do not use outside knowledge, even for well known facts.
- Answer exactly what was asked: the same year, month, group and unit. Never use a value from another year, month or
  group in its place, and never estimate or convert a value the context does not show.
- If the context shows related information in a different form, for example a percent difference and not an amount,
  say that the asked form is not given, then give what is shown. Check what a chart's axis really measures.
- To read a chart, first find the exact position of the asked year or group on the horizontal axis, using the tick
  marks and grid lines. Then use the legend to find the right line or bar at that position. Then read its value
  against the grid lines of the vertical axis. Do not take a value from a neighboring point. Say "about" for any
  value you read from the position of a line or bar.
- End the answer with its source in brackets, using the exact labels given in the context, for example
  [noaa_sea_ice_2020, page 4] or [noaa_sea_ice_2020, Figure 2]. Never cite a page or figure that is not in the context.
- Keep the answer short.
- If the context does not contain the answer or anything related, reply with exactly this sentence and nothing else:
  {not_found}"""


@dataclass
class Answer:
    text: str
    input_tokens: int
    output_tokens: int


def image_as_data_url(path):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_content(question, retrieved, detail):
    """The user message: the question, then the pages, then each figure with its caption and notes."""
    content = [{"type": "input_text", "text": f"Question: {question}\n\nText passages:"}]
    for page in retrieved.pages:
        content.append({"type": "input_text", "text": f"[{page['doc']}, page {page['page']}] {page['text']}"})
    for figure in retrieved.figures:
        label = f"[{figure['doc']}, Figure {figure['number']}] (on page {figure['page']}) {figure['caption']}"
        if figure.get("notes"):
            label += f"\nNotes under the figure: {figure['notes']}"
        content.append({"type": "input_text", "text": label})
        content.append({"type": "input_image", "image_url": image_as_data_url(figure["image_path"]), "detail": detail})
    return content


def unknown_sources(answer, retrieved):
    """Sources the answer cites that were not in the context. The model can invent these, so the code checks."""
    figures = {(figure["doc"], figure["number"]) for figure in retrieved.figures}
    pages = {(page["doc"], page["page"]) for page in retrieved.pages} | {(figure["doc"], figure["page"]) for figure in retrieved.figures}
    unknown = []
    for source in re.findall(r"\[([^\]]+)\]", answer):
        doc = source.split(",")[0].strip()
        cited = [(figures, int(number)) for number in re.findall(r"Figure\s*(\d+)", source)]
        cited += [(pages, int(number)) for number in re.findall(r"page\s*(\d+)", source)]
        if any((doc, number) not in sent for sent, number in cited):
            unknown.append(source)
    return unknown


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
        extra = {"reasoning": {"effort": s.openai_reasoning_effort}} if s.openai_reasoning_effort else {}
        try:
            response = self.client.responses.create(
                model=s.openai_model,
                instructions=INSTRUCTIONS.format(not_found=s.not_found_sentence),
                input=[{"role": "user", "content": build_content(question, retrieved, s.image_detail)}],
                max_output_tokens=s.answer_max_tokens,
                **extra,
            )
        except OpenAIError as exc:
            log.error("the model call failed: %s", type(exc).__name__)
            raise RuntimeError(f"The answer could not be generated: {exc}") from exc
        return Answer(response.output_text.strip(), response.usage.input_tokens, response.usage.output_tokens)
