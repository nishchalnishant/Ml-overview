# Interview 20 — Multimodal LLM Game Testing Agent
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the QA Automation team. Testing UI and menus in modern games (like FIFA or Madden) is a massive manual effort. Every time a button moves slightly, traditional script-based automation (like Selenium/Appium) breaks because it relies on exact X/Y coordinates or strict element IDs.

Your task is to **design an AI Agent using a Multimodal Vision-Language Model (VLM)** that can visually look at the game screen, read natural language instructions (e.g., "Buy a gold pack in the store"), and automatically navigate the UI to execute the test.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Interaction mechanism (How does the AI actually "click" a button?)
- Action Space mapping (How does the VLM know where on the screen a button is?)
- Latency & Cost (GPT-4V costs $0.02 per image. Can we run this at 60 FPS?)
- State Management (What happens if the game is loading or animating?)
- Verification (How does the agent know it succeeded?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"How do we map the VLM's output to actual controller/mouse inputs?"**
   → *Answer: We have a wrapper that accepts `click(x, y)` or `press_button('A')`.*

2. **"How does the model output X/Y coordinates? VLMs are notoriously bad at exact spatial coordinates."**
   → *Answer: Good insight. You must design a system that bridges the gap between the image and exact coordinates (e.g., Set-of-Mark prompting).*

3. **"Is this running in real-time gameplay, or just turn-based UI menus?"**
   → *Answer: Just UI menus. 1 frame per second is fine.*

4. **"Can we use internal game engine hooks to get the UI tree, or must it be purely visual?"**
   → *Answer: We want it to be purely visual (pixel-in, action-out) so it works on console streaming (Xbox Cloud/PS Remote Play) without needing debug builds.*

---

## Part 4 — Expected Assumptions

- **Architecture:** ReAct (Reasoning + Acting) Agent loop. 
- **Model:** A Vision-Language Model (GPT-4o, Claude 3.5 Sonnet, or open-source Qwen-VL).
- **Spatial Grounding:** Use an object detection model (like Grounding DINO or YOLO) to draw bounding boxes on the screen, overlay numeric labels, and pass the *annotated* image to the VLM (Set-of-Mark technique).

---

## Part 5 — High-Level Solution

```
  [Game Screen Stream]
       │ (1 Screenshot)
       ▼
  [Spatial Grounding Module (e.g., Grounding DINO / OpenCV)]
  ┌────────────────────────────────────────────────────────┐
  │ Detects all text and buttons.                          │
  │ Draws a red box with a number (1, 2, 3) over each one. │
  └────────────────────────────────────────────────────────┘
       │ (Annotated Image)
       ▼
  [Multimodal LLM Agent (GPT-4o)]
  Prompt: "Goal: Buy a gold pack. Look at the numbered boxes. 
           Output a JSON reasoning step and the box number to click."
       │
       ▼ (Action: Click Box 4)
  [Execution Engine] ➔ Converts Box 4 to X/Y ➔ Sends Click to Game
       │
       ▼ (Wait for screen transition)
  [Loop Repeats until Goal is met]
```

**Core ML Component:** The "Set-of-Mark" (SoM) or visual-prompting pipeline. Without bounding boxes, a VLM cannot reliably output exact `x, y` coordinates. By overlaying numbers on the image, the VLM just has to output an integer, which is trivial.

---

## Part 6 — Step-by-Step Implementation

### Step 1: UI Element Extraction
- Capture screenshot.
- Run an Optical Character Recognition (OCR) tool (like Tesseract/EasyOCR) and a UI element detector (YOLO) to find all interactable elements.
- Create a dictionary: `{ "1": {"x": 100, "y": 200, "w": 50, "h": 20}, "2": ...}`

### Step 2: Image Annotation
- Using Python (PIL/OpenCV), draw a brightly colored bounding box and a distinct numeric label (1, 2, 3...) over every detected element on the screenshot.

### Step 3: Agent Prompting (ReAct)
- Send the annotated image and the Goal to the VLM.
- Enforce strict JSON output:
  ```json
  {
    "thought": "I am on the Main Menu. The goal is to buy a pack. I see the 'Store' button labeled as 5.",
    "action_type": "CLICK",
    "target_id": "5"
  }
  ```

### Step 4: Execution & Verification
- Parse JSON, map `target_id: 5` back to its X/Y coordinate from Step 1.
- Send OS-level click.
- Wait 2 seconds. Take a new screenshot. Ask the VLM: "Did we achieve the goal?"

---

## Part 7 — Complete Python Code

```python
"""
vlm_qa_agent.py - Visual AI Agent for Game Testing
"""
import logging
import json
import base64
from typing import Dict, Any
# Mock imports for visual processing and LLM
import cv2
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual Processing (Set-of-Mark)
# ---------------------------------------------------------------------------
def extract_and_annotate_ui(image_path: str) -> tuple[str, Dict]:
    """
    Mocks detecting UI elements, annotating the image with numbers,
    and returning the base64 image + coordinate mapping.
    """
    # In reality: Run EasyOCR + YOLO here
    element_map = {
        "1": {"text": "Play", "x": 100, "y": 200},
        "2": {"text": "Store", "x": 100, "y": 300},
        "3": {"text": "Settings", "x": 100, "y": 400}
    }
    
    # In reality: cv2.rectangle() and cv2.putText() to draw boxes/numbers
    annotated_image_path = "annotated_screen.jpg"
    
    with open(annotated_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    return base64_image, element_map

# ---------------------------------------------------------------------------
# VLM Agent Logic
# ---------------------------------------------------------------------------
client = openai.OpenAI(api_key="mock_key")

def run_agent_step(goal: str, base64_image: str) -> dict:
    """Calls the VLM to decide the next action based on the annotated image."""
    
    prompt = f"""
    You are an automated game tester. 
    Your goal is: {goal}
    
    Look at the provided screenshot. The interactable UI elements have been highlighted 
    with a red bounding box and a numeric ID.
    
    Determine your next step to achieve the goal.
    Output ONLY valid JSON in this format:
    {{
        "thought": "Reasoning about current screen and goal",
        "action": "CLICK",
        "target_id": "Number of the box to click (or null if finished)",
        "status": "IN_PROGRESS or SUCCESS or FAILED"
    }}
    """
    
    # Note: Using mock chat completion structure for GPT-4V
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        response_format={ "type": "json_object" }
    )
    
    return json.loads(response.choices[0].message.content)

# ---------------------------------------------------------------------------
# Execution Loop
# ---------------------------------------------------------------------------
def execute_click(x: int, y: int):
    logger.info(f"OS ACTION: Clicking at X:{x}, Y:{y}")
    # pyautogui.click(x, y)

def run_test_scenario(goal: str, max_steps: int = 10):
    logger.info(f"Starting Test Scenario: {goal}")
    
    for step in range(max_steps):
        # 1. Capture & Annotate Screen
        logger.info(f"--- Step {step+1} ---")
        img_b64, coord_map = extract_and_annotate_ui("current_screen.png")
        
        # 2. VLM Decides Action
        decision = run_agent_step(goal, img_b64)
        logger.info(f"Agent Thought: {decision['thought']}")
        
        # 3. Handle Status
        if decision["status"] == "SUCCESS":
            logger.info("Test Passed!")
            return True
        elif decision["status"] == "FAILED":
            logger.error("Test Failed (Agent got stuck).")
            return False
            
        # 4. Execute Action
        if decision["action"] == "CLICK":
            target = decision["target_id"]
            if target in coord_map:
                execute_click(coord_map[target]["x"], coord_map[target]["y"])
            else:
                logger.error(f"Agent hallucinated invalid target ID: {target}")
                return False
                
        # Wait for game transition
        import time
        time.sleep(2)
        
    logger.warning("Max steps reached. Test Timeout.")
    return False

if __name__ == "__main__":
    # run_test_scenario("Navigate to the Store and buy a Gold Pack")
    pass
```

---

## Part 8 — Deployment

### Infrastructure
- **Agent Server:** A Python worker running on a Kubernetes cluster.
- **Game Node:** A physical console (PS5/Xbox) or a Windows PC running the game client.
- The Agent Server connects to the Game Node via a Remote Desktop protocol (e.g., Parsec, WebRTC) to grab video frames and send OS-level USB inputs.

### Cost Control
- GPT-4V costs ~$0.02 per image. A 10-step test costs $0.20. Running 10,000 regression tests a night costs $2,000.
- **Deployment fix:** Train a smaller, open-source local VLM (like Qwen-VL-Chat 7B or LLaVA) on historical test runs and deploy it on local T4 GPUs to drop inference costs to near zero.

---

## Part 9 — Unit Testing

```python
import pytest
from vlm_qa_agent import run_agent_step

# Mock the OpenAI API
class MockResponse:
    class Choice:
        class Message:
            content = '{"thought": "I see the store", "action": "CLICK", "target_id": "2", "status": "IN_PROGRESS"}'
        message = Message()
    choices = [Choice()]

@pytest.fixture
def mock_openai(mocker):
    mocker.patch('vlm_qa_agent.client.chat.completions.create', return_value=MockResponse())

def test_agent_json_parsing(mock_openai):
    # Ensure the agent strictly parses the JSON return format
    result = run_agent_step("Buy a pack", "fake_b64_img")
    
    assert result["action"] == "CLICK"
    assert result["target_id"] == "2"
```

---

## Part 10 — Integration Testing

- Set up a mock "Web Game Menu" in HTML/JS.
- Provide the agent with the goal: "Click the Blue Button".
- Run the full pipeline. The agent should grab the Chrome window screenshot, OCR the screen, annotate the Blue Button, send it to the VLM, get the coordinates, and trigger `pyautogui` to click it.
- Assert that the Web Game Menu registers the click event.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **Animation Latency** | A VLM taking a screenshot during a screen wipe transition will hallucinate. We must add a "Stable Frame" detector (e.g., comparing pixel differences between frames) and only trigger the VLM when the screen has been static for > 500ms. |
| **Error Recovery** | The VLM clicks the wrong button and enters a weird menu. The ReAct loop allows the agent to look at the new screen, realize its mistake, and look for a "Back" button automatically, unlike strict Selenium scripts which just crash. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| Visual Testing vs Internal Hooks | Visual testing (Pixel-in, Mouse-out) is universally compatible with any engine (Frostbite, Unreal) and works on cloud gaming. Tradeoff: It requires heavy ML compute and OCR. Internal engine hooks (extracting the UI tree as JSON) is 100x faster and cheaper, but breaks constantly as engine versions change. |
| API Models vs Local Models | GPT-4o provides perfect out-of-the-box reasoning but is expensive. Local LLaVA is free to run, but has lower zero-shot reasoning capability and requires fine-tuning on game-specific UIs. |

---

## Part 13 — Alternative Approaches

1. **Reinforcement Learning (RL):** Train an RL agent (PPO) to navigate the UI by giving it a reward when it reaches the target screen. Extremely robust, but takes millions of steps to train for *each specific menu flow*. VLMs are zero-shot and require no training.
2. **Template Matching (OpenCV):** The old-school way. Save a `.png` of the "Store" button. Use `cv2.matchTemplate()` to find it on screen. Fails instantly if the art team changes the button color from Blue to Red.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| OCR Failure | The font is stylized (e.g., sci-fi font in Mass Effect). OCR returns gibberish. The VLM doesn't know what the buttons do. | Fall back to Icon/Feature matching. Grounding DINO can detect buttons based on shapes and borders even if the text is unreadable. Pass the cropped button image to the VLM separately. |
| Infinite Loops | Agent clicks a button that opens a popup. It closes the popup. It clicks the button again. | Add a "Memory" array to the prompt. Pass the last 3 actions into the context window: `Previous actions: [Clicked Box 2, Clicked Box 5]`. This breaks loops. |

---

## Part 15 — Debugging

**Symptom:** The agent successfully completes the task (buys a pack), but reports "FAILED".

**Debugging steps:**
1. Check the VLM's final `thought` trace.
2. The VLM says: "I see the pack opening animation, but I do not see a 'Success' message, so I failed."
3. **Fix:** The prompt's definition of the goal was too ambiguous. Update the prompt to define exact success criteria: "Goal: Buy a pack. Success is defined as seeing the Pack Opening Animation on screen."

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `test_flakiness_rate` | > 10% → The VLM is hallucinating coordinates. Needs better image annotation. |
| `average_steps_to_goal` | Spikes unexpectedly → The UI changed and the agent is taking a longer, confusing path to solve the puzzle. |
| `token_cost_per_run` | > $0.50 → Too many screenshots being sent. Reduce resolution. |

---

## Part 17 — Production Improvements

1. **Action Trajectory Caching:** VLMs are slow. Once the VLM figures out the sequence to buy a pack (`Click Store -> Click Gold -> Click Buy`), save this exact sequence of clicks as a standard script. Run the fast script every night. Only invoke the slow VLM agent if the fast script breaks (self-healing UI tests).
2. **DOM-based Overlay:** If we *do* have engine hooks, extract the UI DOM, use it to draw the bounding boxes perfectly (skipping the slow/flaky YOLO step), and just use the VLM for the reasoning step.

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"The UI has a scrollbar. The 'Gold Pack' is not visible on the initial screen. How does your agent know to scroll down to find it?"**
2. **"We have 10,000 UI tests to run every night. Calling OpenAI 10,000 times a night is a massive security risk, as we are uploading unreleased game screenshots to the cloud. How do you mitigate this?"**
3. **"In your Set-of-Mark approach, what happens if there are 150 interactable items on screen (like a massive inventory)? The image becomes a tangled mess of red boxes and numbers."**

---

## Part 19 — Ideal Answers

**Q1 (Scrolling):**
> "We must add 'SCROLL_DOWN' and 'SCROLL_UP' to the agent's action space. We update the prompt: 'If you do not see the target, you may choose to SCROLL'. If the OCR doesn't find the keyword, the VLM's ReAct logic will output `{action: SCROLL}`. The execution engine sends a D-pad down command, takes a new screenshot, and loops."

**Q2 (Security / IP Protection):**
> "We cannot send unreleased IP to public cloud endpoints. We must host the VLM internally. We can deploy a 7B or 13B parameter VLM (like LLaVA or Qwen-VL) on internal EA GPU clusters (e.g., using vLLM for high throughput). Since the Set-of-Mark technique simplifies the reasoning task heavily, a 7B local model performs just as well as GPT-4 for this specific use case."

**Q3 (Dense UI / Bounding Box Clutter):**
> "If there are 150 items, the boxes overlap, destroying the visual context. We solve this via a Two-Pass approach. 
> 1. Pass 1: No bounding boxes. Ask the VLM to output the rough quadrant or region of interest (e.g., 'Top Right Inventory Grid').
> 2. Pass 2: We crop the image to that specific quadrant, run the annotation logic, and pass the cleanly numbered zoomed-in crop back to the VLM for the precise click."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Anticipates the spatial grounding issue (VLMs can't output exact X/Y).
- Proposes a visual prompting technique like Set-of-Mark to bridge the gap.
- Solves the scrolling and dense UI problems elegantly.
- Understands the security implications of uploading unreleased screenshots to OpenAI.

### Hire
- Sets up a solid ReAct agent loop.
- Uses OCR to find buttons.
- Writes clean JSON parsing and OS execution code.
- Understands how to break infinite loops.

### Lean Hire
- Suggests using an LLM to read the internal game memory (requires heavy C++ engine modifications) instead of solving the visual problem requested.
- Has no idea how much a VLM API call costs.

### Lean No Hire
- Proposes standard Selenium/Appium automation and completely ignores the VLM requirement.
- Fails to explain how the AI actually clicks a button on a physical screen.

### No Hire
- Does not know what a Multimodal model is.
- Thinks ChatGPT can natively click buttons on a user's computer without an execution wrapper.
