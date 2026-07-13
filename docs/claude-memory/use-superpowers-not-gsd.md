---
name: use-superpowers-not-gsd
description: "For the ml-lab-template repo, use the superpowers workflow/skills, not GSD"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3c88f287-25a8-4547-8e79-fae937cdda1d
---

For the ml-lab-template repo, use the superpowers plugin skills (brainstorming, writing-plans, executing-plans, TDD, etc.) exclusively. Do NOT use GSD (gsd-*) skills or workflows here.

**Why:** The user explicitly stated this preference at the start of work on this repo (2026-07-10).

**How to apply:** When a task could be handled by either a `gsd-*` skill or a `superpowers:*` skill, always choose the superpowers one. Never invoke `gsd-*` skills in this repo.
