<div align="center">

# ARTHAGATI

**अर्थगति**

Valuation-Anchored Sentiment · Ornstein-Uhlenbeck · Kalman · Walk-Forward Spearman

[![Version](https://img.shields.io/badge/version-2.11.22-2B5FD9)](CHANGELOG.md) [![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](requirements.txt) [![Streamlit](https://img.shields.io/badge/streamlit-1.52.2-FF4B4B)](requirements.txt) [![License](https://img.shields.io/badge/license-proprietary-6B7280)](LICENSE) [![Pragyam Family](https://img.shields.io/badge/pragyam-family-0F1217)](#the-pragyam-family)

</div>

> One score prices market sentiment against its own history, a second instrument says whether
> to believe it, and a held-out test says whether either has ever been worth anything.

## Overview

Arthagati answers one question: **what is the market's sentiment state, how far should that
reading be trusted, and has it ever been worth anything?** It reads macro, breadth and valuation
series from a Google Sheet and scores the market against its own history, anchored to the PE
ratio and Earnings Yield. An independent oscillator says whether to believe the score, and a
held-out test (a 25% holdout with an embargo, against a permutation null and against the plain
PE ratio) says whether either has carried real information.

## What It Does

It ingests macro, breadth, and valuation data from a Google Sheet and
produces five outputs:

| Output | Range | Description |
|--------|-------|-------------|
| **Mood Score** | −100 to +100 | Correlation-weighted composite anchored to PE and Earnings Yield |
| **MSF Spread** | −10 to +10 | Momentum / Structure / Flow / Regime confirmation oscillator |
| **WaveTrend** | (unbounded) | LazyBear oscillator on Mood Score with WT1/WT2 crossover signals |
| **Similar Periods** | — | Historical analogs matched by Mahalanobis distance + trajectory shape, with forward returns at 5D / 20D / 60D / 90D |
| **Predictor Assessment** | — | Transparency into which variables drive the score and which are noise |

## Quick Start

### Local

```bash
# 1. Set environment variables with your Sheet coordinates
export ARTHAGATI_SHEET_ID="<your-spreadsheet-id>"
export ARTHAGATI_SHEET_GID="<your-worksheet-gid>"

# 2. Make sure the sheet is "Anyone with the link can view"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run arthagati.py
```

### Streamlit Cloud

1. Push repo to GitHub
2. **App Settings → Environment Variables** — add `ARTHAGATI_SHEET_ID` and `ARTHAGATI_SHEET_GID`
3. Deploy

## Project Structure

```
arthagati.py        Streamlit app: ingestion, the engine pipeline, the pages
config.py           every tunable constant, the predictor profiles and their measurements
validation.py       the held-out signal validation (holdout, embargo, permutation null)
core/               the family console (logger_config.py)
ui/                 theme.css + theme.py (the family design system), components, charts,
                    panel, signals, format
ui/tabs/            one module per page and panel: landing, mood, analogs, drivers,
                    validation, configuration, and the panels they share
docs/               the guide and the product vision
```

## Documentation

| Document | What it covers |
|---|---|
| [Guide](docs/GUIDE.md) | The system architecture, its mathematics, the data schema, configuration and predictor profiles, features |
| [Vision](docs/VISION.md) | The curation philosophy: which theories were adopted, which were cut, and why |
| [Changelog](CHANGELOG.md) | Every release since v1.2.0, newest first |

## The Pragyam Family

Ten Streamlit applications built on one design system and one set of components, so a
panel, a table, a chart or a number reads the same in each.

| Product | What it is |
|---|---|
| [**Tattva**](https://github.com/thebullishvalue/Tattva) | Cross-asset fair value, with a call only where its own record supports one |
| [**Nishkarsh**](https://github.com/thebullishvalue/Nishkarsh) | Nifty 50 inference: a top-down forecast and bottom-up breadth, read as one |
| **Arthagati** · *this repository* | Valuation-anchored market sentiment, tested on data it never saw |
| [**Vyapti**](https://github.com/thebullishvalue/Vyapti) | Market breadth: where participation stands, which way it moves, and what that has meant |
| [**Pragyam**](https://github.com/thebullishvalue/Pragyam) | Portfolio curation that spreads risk rather than predicting returns |
| [**Sanket**](https://github.com/thebullishvalue/Sanket) | A market signal screener: what fired, what is about to, and whether it has paid |
| [**Vaaydo**](https://github.com/thebullishvalue/Vaaydo) | An F&O trade screener on option positioning, with futures tickets |
| [**Nivesa**](https://github.com/thebullishvalue/Nivesa) | A bond portfolio ledger: cost, yield, duration and cashflows across accounts |
| [**Samhita**](https://github.com/thebullishvalue/Samhita) | A portfolio tracker: live prices, performance and risk against the benchmark |
| [**Swing**](https://github.com/thebullishvalue/Swing) | A portfolio tracker for the swing book, on the same engine as Samhita |

## License

Proprietary. © 2026 @thebullishvalue. All rights reserved. The source is visible here
for reference only: viewing it grants no right to use, copy, modify or distribute it.
See [LICENSE](LICENSE).

## Disclaimer

Arthagati is a research and decision-support tool, not investment advice. Nothing it shows is
a recommendation to buy or sell any security, and past or measured results do not guarantee
future ones. Every decision made with it is your own.

---

<div align="center"><sub>© 2026 @thebullishvalue · Part of the Pragyam product family</sub></div>
