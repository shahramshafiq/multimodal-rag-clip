# Multimodal RAG on PDF Reports

Ask questions about two short PDF reports and get answers from their text and from their charts and maps, with the source named. Search finds the right pages and figures, and a vision model reads the real pictures.

## How it works

```text
ONCE, FOR EACH PDF
PDF --> text blocks --> chunks --> text embeddings ----------------------+
    |                                                                    +--> index
    +-> figures cropped through their captions                           |
           +--> a vision model describes each figure --> text embeddings +
           +--> CLIP image vectors (kept for comparison) ----------------+

FOR EACH QUESTION
question --> text embedding --> top 6 chunks --> their whole pages --+
                             +-> top 2 figure descriptions           +--> vision model --> answer + source
                                    --> the real pictures -----------+
```

Five design choices, each for a reason:

1. **Figures are found through captions, not through the images stored in the PDF.** The CDC charts appear to be drawn as vector shapes, so a plain image extract could miss them. Logos and QR codes have no caption, so they are left out with no special rule.
2. **Figures are searched through a text description, and the answer reads the real picture.** A vision model describes each figure once. The description is only used to find the figure, so a mistake in it can never become a wrong number in an answer.
3. **Text and figures are ranked in two separate lists,** because their scores are not on the same scale.
4. **Chunks are for searching, whole pages are for reading.** A chunk is often cut off from its context, so the model gets the full page of each chunk found.
5. **The model must not fill gaps.** It may not use another year's value or outside knowledge, and it must say when the asked form is not given. The code also checks that every source an answer cites was really sent to the model.

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
  chunking.py           Cuts the text into chunks
  embeddings.py         CLIP image vectors and OpenAI text embeddings
  describe.py           One search description per figure, saved after the first run
  store.py              Saves, loads and searches the vectors with NumPy
  retrieve.py           Separate text and figure search, whole pages, rank fusion
  answer.py             Sends pages and pictures to the vision model, checks sources
  evaluation.py         Retrieval and answer scoring
data/pdfs/              The two reports
data/figures/           The cropped figures
data/figure_descriptions.json   The saved figure descriptions
eval/golden.json        71 questions with expected sources and answers
eval/results/           Saved results
tests/                  18 tests, no key and no CLIP download needed
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
python ask.py "What is the percent difference in March 2010 and in September 2010?"
python ask.py "bar chart of Long COVID by race" --no-llm
python evaluate.py
python evaluate.py --answers
python evaluate.py --answers --compare-text-only
python -m pytest
```

Each question stands alone: `ask.py` does not remember the question before it. A question costs about half a cent. `--no-llm` and `evaluate.py` only use the embedding model, which costs a tiny fraction of that. The tests cost nothing. The first `ingest.py` run downloads CLIP and pays once for the figure descriptions.

## Evaluation

The golden dataset has 71 questions in three sets. Every chart value was measured from the chart's pixels, so the true answers do not depend on any model.

- **Easy (28):** worded almost like the captions and the report.
- **Natural (27):** worded the way a person asks, including 5 traps where the documents do not hold the value and a model could easily invent it.
- **Holdout (16):** added after every setting was chosen, and never tuned on.

**Answers, same pipeline with and without figures:**

| Set and question type | With figures | Figures removed |
|---|---|---|
| Easy: text | 12/12 | 12/12 |
| Easy: figure | 12/12 | 2/12 |
| Easy: not in the documents | 4/4 | 4/4 |
| Natural: text | 8/8 | 8/8 |
| Natural: figure | 13/14 | 2/14 |
| Natural: trap | 5/5 | 5/5 |
| Holdout: text | 4/4 | 4/4 |
| Holdout: figure | 10/10 | 2/10 |
| Holdout: not in the documents | 2/2 | 2/2 |
| **Total** | **70/71** | **41/71** |

No answer cited a source that was not sent. All 36 figure questions were run twice, and no question changed between right and wrong. Reading the answers by hand, chart values are good estimates, not exact: one run read 2010 first-year ice as 65% and the next as 70%, against a measured value of about 65%.

The one miss is `natural_fig_06`: the right figure was not retrieved, so the model said the exact amount is not given. It invented nothing.

**The first version on the same questions** (CLIP for everything, `gpt-4.1-mini`) scored natural figure 7/14 and traps 2/5 on a stricter trap check, invented sources, and used a 2020 value for a 2010 question.

**Figure search, right figure found on natural questions:**

| Search by | Top 1 | Top 2 |
|---|---|---|
| CLIP on the image | 4/14 | 8/14 |
| Figure description (used) | 11/14 | 13/14 |
| Both, with rank fusion | 5/14 | 13/14 |

On the holdout set the same three scored 5/10, 9/10 and 7/10 in the top 1.

**Vision model, the 20 hardest questions with the same retrieval:**

| Model | Correct |
|---|---|
| `gpt-5.4-mini` (used) | 19/20 |
| `gpt-5.4` | 18/20 |
| `gpt-4.1-mini` | 14/20 |

## What the measurements changed

- The first golden questions were too close to the captions, so they scored well while real questions failed. Natural questions and a holdout set were added for that reason.
- CLIP lost to text search over figure descriptions, so CLIP is no longer used to answer questions. It is still built into the index, and `FIGURE_SEARCH=both` or `image` brings it back.
- A dedicated text embedding model replaced CLIP for text. The right page in the top 1 went from 2/8 to 6/8 on natural text questions.
- The bigger model was not the better one, so the smaller one is used.
- The scorer itself was wrong several times: the 5 in "Figure 5" passed as an answer, and so did any answer naming the right word in a wrong claim. So answers are also read by hand.

## Limitations

- 71 questions on 2 documents is small, and one person wrote them, so these numbers are not a general claim. Only the 16 holdout questions were never tuned on.
- Values read from a chart are estimates, so a small error is allowed when scoring them.
- A figure without a caption is not found.
- The saved descriptions were written by `gpt-4.1-mini`, before the answer model was changed. They have only been used for search.
- The first question in a terminal is slow, about 17 seconds in one test, partly from loading the libraries.

## Author

**Shahram Shafiq** - [GitHub](https://github.com/shahramshafiq) | [LinkedIn](https://www.linkedin.com/in/shahramshafiq/)
