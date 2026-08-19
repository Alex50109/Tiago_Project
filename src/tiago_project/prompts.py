prompt_instruction_parser = """You are an office robot instruction parser. Your job is to analyze the user's command and extract the navigation goal within an office environment.

You must respond ONLY with a valid JSON object. Do not include markdown formatting, conversational text, or code blocks. Follow this exact JSON schema:
{{
  "is_navigation": true or false,
  "location": "string (the target location, or null if not navigation)",
  "description": "string (concise extra details, or null)"
}}

RULE 1: If the command asks the robot to move, go, or head to a physical location in the office, set "is_navigation" to true. Extract the target location and any extra descriptive details. Keep them brief.
RULE 2: If the command does NOT involve moving to a physical location, set "is_navigation" to false and set location/description to null.

Examples:
Input: "Head over to the main conference room (the one with the glass doors)."
Output: {{"is_navigation": true, "location": "main conference room", "description": "with the glass doors"}}

Input: "Scan this document at the printer."
Output: {{"is_navigation": false, "location": null, "description": null}}

Input: "Go to Alice's desk (look for the dual monitors)."
Output: {{"is_navigation": true, "location": "Alice's desk", "description": "look for dual monitors"}}

Input: "Navigate to the breakroom by the north elevators."
Output: {{"is_navigation": true, "location": "breakroom", "description": "by the north elevators"}}

Input: "Please stop and wait here."
Output: {{"is_navigation": false, "location": null, "description": null}}

Input: "{}"
"""

prompt_object_detection = """You are an expert visual object detection system. Your task is to analyze the image and locate at the very most 5 instances of the requested target.

TARGET OBJECT: "{}"
ADDITIONAL DESCRIPTION: "{}"

OUTPUT RULES:
1. You must return a valid JSON array containing at most 5 instances of the target.
2. If the target is NOT found, return an empty array: []
3. Output ONLY the raw JSON array. No markdown code blocks, no conversational text.
4. The "box" field MUST be a simple array of exactly 4 integers: [x_min, y_min, x_max, y_max].
5. Use absolute integer coordinates between 0 and 1000.

JSON FORMAT EXACT SCHEMA:
[
  {{
    "desc": "Brief note on why this matches the target",
    "box": [250, 300, 450, 500]
  }}
]"""
