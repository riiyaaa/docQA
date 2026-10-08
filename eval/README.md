# Evaluation dataset

A fictional document set with 40 questions whose answers are known, used to
measure how well DocQA retrieves and answers.

**Quillmere Appliances is an invented company.** Every product, policy,
phone number, and email address in `sample_docs/` is made up.

## The documents (`sample_docs/`)

| File | Format | Contents |
|---|---|---|
| `warranty_guide.pdf` | PDF, 5 pages | Consumer warranty: coverage, exclusions, claims, refunds |
| `pro_commercial_warranty.pdf` | PDF, 2 pages | Commercial warranty with different terms (distractor) |
| `ap300_manual.pdf` | PDF, 4 pages | AP300 air purifier: specs, operation, maintenance, error codes |
| `ap200_manual.pdf` | PDF, 3 pages | AP200 purifier: same error codes, different meanings (distractor) |
| `rf500_manual.pdf` | PDF, 4 pages | Refrigerator manual |
| `dw700_manual.pdf` | PDF, 3 pages | Dishwasher manual |
| `support_faq.docx` | Word | Support hours, shipping, extended warranty, refurbished items |
| `store_policies.md` | Markdown | Returns, shipping, price adjustments |
| `ap300_release_notes.txt` | Text | AP300 firmware versions |

The distractors are deliberate. The same error codes (E04, E11, E17, E21)
mean different things on different products, part numbers look alike
(ZX-4410 vs. ZX-4401, AP-F300 vs. AP-F200), and the two warranties have
different terms. Retrieval has to pick the right document, not just a
plausible one.

## The questions (`questions.jsonl`)

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | `q01` to `q40` |
| `type` | `easy`, `exact_term`, `reworded`, `multi_part`, or `unanswerable` |
| `question` | What the user asks |
| `answer` | A correct reference answer |
| `answer_keywords` | Terms a correct answer should mention. `\|` separates alternatives: `"2 years\|two years"` |
| `sources` | Where the answer lives: `{"source": file, "page": n}`. `page` is `null` for files without pages |
| `evidence` | Exact phrases from those sources that support the answer |

Question types:

- **easy (11):** stated plainly, in words close to the document's.
- **exact_term (9):** hinge on a code or part number. Tests keyword search.
- **reworded (8):** asked in different words than the document uses. Tests vector search.
- **multi_part (6):** need two facts, a small calculation, or two documents.
- **unanswerable (6):** not in the documents. The correct reply is exactly
  `I couldn't find that in the documents.`

`answer_keywords` is a quick, rough check. A correct answer can use other
wording, so the Phase 2 evaluation also uses an LLM judge.

## Usage

Index the sample documents (delete `chroma_db` first so nothing else is mixed in):

```bash
python cli.py ingest sample_docs
python cli.py ask "What does error code E17 mean on the AP300?"
```

After editing the documents or the questions, confirm they still match:

```bash
python eval/check_dataset.py
```

To change the documents, edit `scripts/make_sample_docs.py` and regenerate
(requires `pip install reportlab`):

```bash
python scripts/make_sample_docs.py
```

## Baseline

Keyword search (BM25) alone puts a correct source page in its top 4 results
for 27 of the 34 answerable questions (79%). Phase 2 measures hybrid search
and later improvements against this.
