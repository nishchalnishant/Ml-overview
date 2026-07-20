---
module: Meta
topic: Overview
subtopic: ""
status: unread
tags: [meta, index]
prerequisites: []
---
# Meta — How To Use This Repo

Navigation, conventions, and tooling for the rest of the repo.

| File | Purpose |
| :--- | :--- |
| `REPO-STRUCTURE.md` | Target layout, design principles, per-file format, migration phases |
| `check-links.py` | Link + `SUMMARY.md` integrity checker. Exit 0 = clean |
| `REMOVED-CONTENT.md` | Audit trail of deleted material |

**Run before every commit:**

```bash
python3 00-meta/check-links.py
```

`SUMMARY.md` is GitBook's navigation source of truth (see `.gitbook.yaml`). Any file move must update it in the same commit, or pages silently vanish from the published site.
