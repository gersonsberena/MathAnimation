# Asset licensing manifest

Every asset referenced from a recipe (music track, font, SVG path source
for engines like `epicycle_fourier`) must have a row here **before** it's
referenced — see README Section 11.4. This is the audit trail if a
licensing question or Content-ID claim comes up later; don't rely on
tribal knowledge of where a file came from.

| Path                    | Source / URL | License | Attribution required? | Added |
|--------------------------|---------------|---------|------------------------|-------|
| _(none committed yet)_   |               |         |                        |       |

## Format

- **Path** — relative path under `assets/`.
- **Source / URL** — where it was obtained.
- **License** — e.g. CC0, CC-BY 4.0, purchased/commercial license (link
  the license terms, not just the marketplace listing).
- **Attribution required?** — yes/no; if yes, note the exact required
  credit text.
- **Added** — date (YYYY-MM-DD).
