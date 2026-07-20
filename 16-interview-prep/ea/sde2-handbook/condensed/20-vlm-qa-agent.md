# Interview 20 — Multimodal VLM Game Testing Agent (Condensed)

Traditional coordinate/ID-based UI automation (Selenium/Appium) breaks whenever a game's UI shifts. Design an AI agent that uses a Vision-Language Model to read a screenshot + natural-language goal ("Buy a gold pack") and drive the game purely visually — no engine hooks.

## Core Architecture
- **Spatial Grounding Module** (Grounding DINO/YOLO/OCR): detects buttons/text, draws numbered bounding boxes on the screenshot — this is the **Set-of-Mark (SoM)** technique, the key ML idea: VLMs can't reliably output pixel coordinates, but they *can* output "click box 4."
- **VLM Agent (ReAct loop)**: GPT-4o/Claude/Qwen-VL takes annotated image + goal, returns strict JSON `{thought, action, target_id, status}`.
- **Execution Engine**: maps `target_id` → real X/Y from the detection dict, sends OS-level click/controller input.
- **Verification step**: re-screenshot after action, ask VLM "did we achieve the goal?" — closes the loop instead of assuming success.
- **Memory buffer**: last N actions fed back into the prompt to break infinite click loops (open/close popup cycles).
- **Stable-frame gate**: only invoke the VLM once frame-diff shows the screen stopped animating (avoids hallucination mid-transition).

## Talking Points That Signal Seniority
- Proactively flags that VLMs cannot output reliable exact coordinates and proposes Set-of-Mark visual prompting to bridge the gap — this is the single most important insight for this problem.
- Raises the security/IP concern unprompted: uploading unreleased game screenshots to a public API (OpenAI) is a data-leak risk; proposes self-hosting a 7B/13B open VLM (Qwen-VL/LLaVA) on internal GPUs via vLLM.
- Notices dense-UI/scroll problems before being asked and proposes a two-pass (coarse region → cropped zoom) annotation strategy for screens with 100+ elements.
- Mentions action-trajectory caching: once the VLM solves a flow once, save it as a fast deterministic script and only fall back to the slow VLM agent when the script breaks (self-healing tests).
- Calls out that a ReAct loop's real advantage over Selenium is graceful error recovery — agent can notice it's on the wrong screen and find "Back" instead of crashing.
- Quantifies cost explicitly (e.g. $0.02/image × 10 steps × 10k tests/night = $2k/night) without being prompted, and ties it to the local-model mitigation.
- Adds a stale/animating-frame guard (pixel-diff stability check) so the agent doesn't act on a mid-transition frame and hallucinate.

## Top 3 Tradeoffs
- **Visual testing vs. engine hooks**: pure vision works across any engine and on cloud streaming, but costs heavy ML compute/OCR; engine hooks (UI tree JSON) are 100x cheaper/faster but break on every engine version bump.
- **API VLM vs. local VLM**: GPT-4o gives best zero-shot reasoning out of the box but is expensive and a data-exfiltration risk; local LLaVA/Qwen-VL is free and private but needs fine-tuning/prompt tuning to match reasoning quality.
- **RL agent vs. VLM agent**: PPO-trained RL is extremely robust once trained but needs millions of steps per menu flow; VLM agents are zero-shot with no training cost, at the price of per-call latency/expense.

## Biggest Pitfall
Proposing standard Selenium/Appium-style automation (or engine-memory hooks) instead of actually solving the visual-grounding problem the question is about — this is an automatic No Hire regardless of how clean the rest of the code is.
