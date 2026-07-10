# Interview 20 — Multimodal VLM Game Testing Agent (Condensed)

Traditional coordinate/ID-based UI automation (Selenium/Appium) breaks whenever a game's UI shifts. Design an AI agent that uses a Vision-Language Model to read a screenshot + natural-language goal ("Buy a gold pack") and drive the game purely visually — no engine hooks.

## Clarifying Questions to Ask
- How does the agent actually "click"? → Wrapper accepts `click(x,y)` / `press_button('A')`.
- How does the VLM get exact X/Y coords (VLMs are bad at this)? → Interviewer wants you to propose a fix (Set-of-Mark).
- Real-time gameplay or turn-based menus? → Just UI menus, 1 FPS is fine.
- Can we use engine/UI-tree hooks instead of pure vision? → No — must be pixel-in/action-out so it works on cloud streaming (Xbox Cloud/PS Remote Play) without debug builds.
- What's the cost/latency budget? → GPT-4V ≈ $0.02/image; 10k regression tests/night gets expensive fast — expect this to be probed.
- How do we know the agent succeeded? → No built-in signal; must be designed (ask VLM to self-verify against a defined success state).

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

## Toughest Follow-ups
**Q: 10,000 UI tests nightly means 10,000 calls to OpenAI with unreleased game screenshots — how do you mitigate the IP/security risk?**
A: Don't send unreleased assets to a public endpoint at all — self-host a 7B–13B VLM (LLaVA/Qwen-VL) on internal GPU clusters, served with vLLM for throughput. Since Set-of-Mark already reduces the task to "pick a numbered box," a small local model performs close to GPT-4 for this narrow use case, so the accuracy loss is acceptable in exchange for eliminating the leak risk.

**Q: What happens with 150 interactable items on screen (e.g., a massive inventory) where bounding boxes overlap into a mess?**
A: Two-pass approach — first pass, no boxes, ask the VLM only for the coarse region of interest ("top-right inventory grid"); second pass, crop to that region, annotate only the now-sparser local elements, and send the zoomed crop back for the precise click. This keeps each VLM call visually simple.

**Q: The target isn't visible on-screen (needs scrolling) — how does the agent know to scroll?**
A: Add `SCROLL_UP`/`SCROLL_DOWN` to the action space and instruct the prompt that scrolling is a valid move when the target isn't found. If OCR/detection can't find the goal keyword on the current frame, the ReAct loop naturally emits a scroll action, the engine sends the input, and the loop re-screenshots and retries.

## Biggest Pitfall
Proposing standard Selenium/Appium-style automation (or engine-memory hooks) instead of actually solving the visual-grounding problem the question is about — this is an automatic No Hire regardless of how clean the rest of the code is.
