# Multimodal RAG with CLIP

Ask questions about two short PDF reports and get answers from their text and from their charts and maps. CLIP searches text and images in one shared vector space, and a vision model reads what was found.

## How it works

```text
PDF --> text blocks --> chunks of at most 70 tokens --> CLIP text vectors --+
    |                                                                       +--> index
    +-> figures, cropped through their captions ------> CLIP image vectors -+

question --> CLIP text vector --> top 8 chunks --> their whole pages -+
                              +-> top 2 figures (pictures) -----------+--> vision model --> answer + source
```

Four design choices, each for a reason:

1. **Figures are found through captions, not through the images stored in the PDF.** The CDC charts are drawn as vector shapes, so a plain image extract would miss them. Logos and QR codes have no caption, so they are left out with no special rule.
2. **Text and figures are ranked in two separate lists.** In CLIP, text-to-text scores run higher than text-to-image scores, so one shared list would push the figures out.
3. **Chunks are for searching, whole pages are for reading.** CLIP reads at most 77 tokens, so chunks are small and lose their context. The model gets the full page of each chunk found.
4. **CLIP finds a chart and the vision model reads it.** CLIP cannot reliably read values on a chart. It only needs to find the right picture.

## Documents

| File | Source | Pages | Figures |
|---|---|---|---|
| `noaa_sea_ice_2020.pdf` | [NOAA Arctic Report Card 2020: Sea Ice](https://repository.library.noaa.gov/view/noaa/27904/noaa_27904_DS1.pdf) | 10 | 5: maps, line chart, area chart, histogram |
| `cdc_long_covid_children_2022.pdf` | [CDC NCHS Data Brief No. 479](https://www.cdc.gov/nchs/data/databriefs/db479.pdf) | 6 | 3: bar charts |

## Project structure

```text
ingest.py               Builds the index from the PDFs
ask.py                  Asks one question
evaluate.py             Scores the pipeline against the golden dataset
multimodal_rag/
  config.py             Reads every setting from .env and .env.example
  extract.py            Text blocks and captioned figures from a PDF
  chunking.py           Chunks that fit CLIP's token limit
  embeddings.py         CLIP text and image vectors
  store.py              Saves, loads and searches the vectors with NumPy
  retrieve.py           Separate text and figure search, whole pages
  answer.py             Sends pages and pictures to the vision model
  evaluation.py         Retrieval and answer scoring
data/pdfs/              The two reports
data/figures/           The cropped figures
eval/golden.json        28 questions with expected sources and answers
eval/results/           Saved results
tests/                  12 tests, no key and no CLIP download needed
```

## Setup and run

Python 3.12. In PowerShell:

```bash
git clone https://github.com/shahramshafiq/multimodal-rag-clip.git
cd multimodal-rag-clip
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Put your key in `.env`: `OPENAI_API_KEY=...`. All other settings, including model names, are read from `.env.example`.

```bash
python ingest.py
python ask.py "In the line chart of monthly sea ice extent, in which year does the September line reach its lowest point?"
python ask.py "bar chart of Long COVID by race" --no-llm
python evaluate.py
python evaluate.py --answers
python -m pytest
```

`ingest.py`, `--no-llm`, `evaluate.py` and the tests run locally and cost nothing. The first run downloads CLIP. `ask.py` without `--no-llm` and `evaluate.py --answers` make paid model calls.

## Evaluation

The golden dataset has 28 questions: 12 answered by text, 12 answered only by looking at a figure, and 4 the documents cannot answer.

**Answers.** The same pipeline was run twice, once with figures removed:

| Question type | Multimodal RAG | Text-only RAG |
|---|---|---|
| Text | 12/12 | 12/12 |
| Figure | 11/12 | 5/12 |
| Not in the documents | 4/4 | 4/4 |
| Total | 27/28 | 21/28 |

The score checks for expected terms in the answer, with sources removed first. Reading the answers by hand shows two things the score cannot:

- The 5 text-only figure scores are too generous. One says the color is yellow and only mentions pink in passing, so it is wrong. Of the other 4, only `figure_08` can be worked out from the text. The rest are guesses, because the model never saw a figure.
- In the multimodal run, `figure_02` is a real error: it says September is below March around 1980, and the chart shows the opposite. `figure_04` names the right group but says "about 60%" where the chart shows about 70%. `figure_08` was answered from the text, because its figure was not retrieved.

**Retrieval**

| Figure search | Top 1 | Top 2 | Top 3 |
|---|---|---|---|
| Image only (used) | 8/12 | 11/12 | 11/12 |
| Image and caption | 6/12 | 9/12 | 10/12 |

| Text search | Top 1 | Top 2 | Top 4 | Top 8 |
|---|---|---|---|---|
| Right page | 8/12 | 9/12 | 10/12 | 12/12 |
| Chunk holding the answer | 7/12 | 7/12 | 8/12 | 9/12 |

What the measurements changed:

- Adding captions to figure search looked like a good idea and made it worse, so it is off.
- With chunks sent to the model, text scored 9/12: the chunk holding the answer ranked as low as 40th of 121. Sending whole pages raised it to 12/12 with CLIP unchanged.

## Limitations

- CLIP is weak at finding exact facts in text. A dedicated text embedding model would very likely search text better, and the two-list design would allow that change.
- A figure without a caption is not found.
- A figure with several panels gets one vector, so one panel is hard to find. `figure_08` missed for this reason.
- 28 questions on 2 documents is small, and the settings were chosen on the same questions they are scored on, so these numbers are not a general claim.
- The model is `gpt-4.1-mini`, which OpenAI's model page marks as deprecated. It worked in these runs and can be changed with `OPENAI_MODEL`.

## Author

**Shahram Shafiq** - [GitHub](https://github.com/shahramshafiq) | [LinkedIn](https://www.linkedin.com/in/shahramshafiq/)
