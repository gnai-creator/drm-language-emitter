# Compliance Checklist

This checklist is informational, not legal advice.

## Repository License

- The repository declares `AGPL-3.0-only` in `pyproject.toml`.
- See `LICENSE` for the AGPL text.
- See `LICENCE-COMMERCIAL.md` for the commercial-license notice.

## Before Using This Project

- Confirm whether your use is compatible with AGPL-3.0-only.
- If you distribute modified versions or provide network access to modified software, review the AGPL source-sharing obligations.
- For closed-source or proprietary use, review `LICENCE-COMMERCIAL.md` and obtain appropriate permission from the rights holder.
- Preserve copyright notices from `COPYRIGHT`.
- Track any datasets or third-party artifacts added to the repo in `docs/third_party_licenses.md`.

## Before Publishing Experiment Artifacts

- Include the exact config file.
- Include the commit hash or source archive used to run the experiment.
- State whether CUDA, CPU, or other accelerators were used.
- State dataset provenance and license.
- Avoid claiming broad model superiority from tiny diagnostic benchmarks.

## Before Adding Data

- Prefer generated, public-domain, or explicitly permissive datasets.
- Add provenance, license, and intended use to `docs/third_party_licenses.md`.
- Do not commit private, personal, credentialed, or unclear-license data.
