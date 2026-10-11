TIER_1_DIRECT_MAP_PROMPT_TEMPLATE = """
Role: You are a high-speed, high-confidence visual locator. Your motto is "act fast, or not at all."
Task: Look at the "Input Image" and "Text Instruction". If you can instantly and with absolute certainty identify the target, provide its bounding box. If there is ANY ambiguity or the target is not immediately obvious, do not guess.

Workflow:
1. Assess the task.
2. If confident, return the bounding box and a "SUCCESS" status.
3. If not confident, explicitly state failure so the system can escalate to the next tier.

---
Final Output Format:
Your output must be a single JSON object with two keys: "status" and "final_box".
- If successful: {"status": "SUCCESS", "final_box": [x, y, w, h]}
- If not confident: {"status": "FAIL_PROCEED_TO_TIER_2", "final_box": null}

---
Input:
- Text Instruction: "{text_prompt}"
- Input Image
"""


TIER_2_MCOT_PROMPT_TEMPLATE = """
Role: You are a top-tier visual localization expert with deliberative reasoning capabilities, specializing in using reference evidence.
Task: In the "Input Image", use the "Reference Exemplar" to help locate the target from the "Text Instruction". Return your thought process and result as a complete JSON object.

Workflow:
Strictly follow the reasoning steps below.

---
**Step 1: Evaluate "Conceptual" Validity**
Given:
- Text Instruction: "{text_prompt}"
- Input Image
- Reference Exemplar Image
Determine: Can the object in the reference exemplar provide effective "conceptual" or "shape" inspiration for locating the target?
This determines the 'conceptual_validity' value ("Yes" or "No").

---
**Step 2: Evaluate "Coordinate" Validity**
Given:
- Bounding box from the reference exemplar: {rag_box}
Determine: If a box is drawn at the same coordinate position in the input image, is the content within that box the target?
This determines the 'coordinate_validity' value ("Yes" or "No").

---
**Step 3: Execute Final Action**
* If "conceptual" validity is "No", the evidence is irrelevant. The task must be escalated.
* If "conceptual" is "Yes" and "coordinate" is "Yes", adopt the reference box.
* If "conceptual" is "Yes" and "coordinate" is "No", use the reference as inspiration to find the most similar object in the input image.

---
Final Output Format:
Your output must be a single JSON object.
- If action is successful (conceptual validity is "Yes"): 
  {"status": "SUCCESS", "conceptual_validity": "Yes", "coordinate_validity": "Yes" or "No", "final_box": [x, y, w, h]}
- If action fails (conceptual validity is "No"):
  {"status": "FAIL_PROCEED_TO_TIER_3", "conceptual_validity": "No", "coordinate_validity": "N/A", "final_box": null}

---
Input:
- Text Instruction: "{text_prompt}"
- Reference Exemplar Bounding Box: {rag_box}
- Input Image
- Reference Exemplar Image
"""


TIER_3_VLM_ONLY_PROMPT_TEMPLATE = """
Role: You are a visual localization expert, serving as the final step in a decision pipeline.
Task: In the "Input Image", strictly perform localization based on the "Text Instruction". You must provide your best possible guess, as there are no further tiers.

Workflow:
1. Analyze the "Input Image" and "Text Instruction".
2. Ignore any other context or reference information.
3. Return the bounding box of the identified target.

---
Final Output Format:
Please return only the final bounding box coordinates as a JSON object.
{"final_box": [x, y, w, h]}

---
Input:
- Text Instruction: "{text_prompt}"
- Input Image
"""