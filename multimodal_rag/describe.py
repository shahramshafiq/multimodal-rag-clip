"""Step 3b: have a vision model describe each figure once, so that text search can find it.

CLIP matches the look of a picture, but it does not know that a chart is about "March and September sea ice". The
description is only used to find the figure. The answer always reads the real picture, so a mistake in a description
can never become a wrong number. Descriptions are saved in a file, so they are paid for only once.
"""
import json
import logging
import os

from openai import OpenAI, OpenAIError

from .answer import image_as_data_url

log = logging.getLogger("describe")

PROMPT = """This figure comes from a report. Its caption is: {caption}

Describe it for a search index, in at most 120 words of plain text. Say what kind of figure it is, what each axis or
color scale shows with its unit and range, each legend entry, every place or group named, and the main pattern.
Do not list individual data values."""


def _describe(client, figure, settings):
    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": PROMPT.format(caption=figure.caption)},
                {"type": "input_image", "image_url": image_as_data_url(figure.image_path), "detail": settings.image_detail},
            ]}],
            max_output_tokens=settings.answer_max_tokens,
        )
    except OpenAIError as exc:
        raise RuntimeError(f"Figure {figure.figure_id} could not be described: {exc}") from exc
    return response.output_text.strip()


def load_descriptions(figures, settings):
    """{figure_id: search text}. Only figures not yet in the file are sent to the model."""
    path = settings.descriptions_file
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [figure for figure in figures if figure.figure_id not in saved]
    if missing:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is empty. Add it to .env in the project folder.")
        client = OpenAI(api_key=api_key, timeout=settings.openai_timeout_seconds, max_retries=settings.openai_max_retries)
        for figure in missing:
            log.info("describing %s", figure.figure_id)
            saved[figure.figure_id] = _describe(client, figure, settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=2, ensure_ascii=True), encoding="utf-8")
    # the caption goes first, because it uses the report's own words
    captions = {figure.figure_id: figure.caption for figure in figures}
    return {figure_id: f"{captions[figure_id]} {saved[figure_id]}" for figure_id in captions}
