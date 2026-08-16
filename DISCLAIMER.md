# Disclaimer

## Data Responsibility

**You are fully responsible for any data you load, process, or export using AutoTabML Studio.**

AutoTabML Studio is a local-first tool — all processing happens on your machine. The
project maintainer has no access to your data, no visibility into what you run, and
no ability to recover, protect, or take responsibility for anything you process with
this software.

Before using the app with sensitive, personal, proprietary, or regulated data:

- Ensure you have the legal right to process that data on your local machine.
- Ensure compliance with any applicable privacy regulations (GDPR, HIPAA, CCPA, etc.).
- Understand that AI-generated model summaries are sent to the LLM provider you
  configure (OpenAI, Anthropic, Gemini, or a local Ollama instance). Raw data rows
  are never included in those requests, but you remain responsible for any
  information derived from your data.
- Foundation model checkpoints (TabFM, TimesFM) carry their own licenses. TabFM
  in particular is non-commercial and research-only — review its license before use.

## No Warranty

This software is provided "as is", without warranty of any kind, express or implied.
The author makes no guarantees about correctness, fitness for a particular purpose,
or suitability for production workloads. Use at your own risk.

See the [MIT License](LICENSE) for the full legal terms.

## Model Outputs

Machine learning models produced by AutoTabML Studio are statistical approximations.
They may be wrong, biased, or overfit. Do not rely on model outputs for decisions
with serious real-world consequences without independent validation by a qualified
professional.

## Third-Party Services

When you supply API keys for OpenAI, Anthropic, or Google Gemini, your usage is
governed by each provider's terms of service. The project has no affiliation with
any of these providers.

## No Financial Support Needed

This project is free and will remain free. The author does not want, need, or accept
financial contributions, sponsorships, or donations of any kind. If you find the
project useful, the best way to contribute is through code, bug reports, or
documentation — see [CONTRIBUTING.md](CONTRIBUTING.md).
