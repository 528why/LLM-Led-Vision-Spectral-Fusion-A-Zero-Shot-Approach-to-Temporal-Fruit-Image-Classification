# LLM_as_expert.py

import os
import json
from openai import OpenAI

# --- Client Initialization ---
# Initialize the OpenAI client.
# It's a best practice to use an environment variable for your API key.
# In your terminal, run: export OPENAI_API_KEY='your_api_key_here'
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    print("Error: The OPENAI_API_KEY environment variable is not set.")
    print("Please set it before running the script.")
    exit()


def extract_knowledge(fruit_name: str) -> str:
    """
    Calls a GPT model to extract the color and texture of a fruit
    in both fresh and stale conditions, then formats it into a specific string.

    Args:
        fruit_name: The name of the fruit (e.g., 'avocado', 'banana').

    Returns:
        A formatted string like:
        "(fresh_color or stale_color) color, (fresh_texture or stale_texture) texture fruit_name"
        Returns an error message if the API call fails or the response is invalid.
    """
    # This prompt is specifically designed to instruct the model to return a JSON object,
    # which is more reliable than parsing plain text.
    prompt = f"""
    What does the {fruits} look likein the photo? It could be either fresh or stale.

    Please provide the output as a single, clean JSON object with the following keys:
    - "fresh_color"
    - "fresh_texture"
    - "stale_color"
    - "stale_texture"
    
    Do not include any text or explanation outside of the JSON object.
    """

    try:
        # We use a newer model like 'gpt-4o' or 'gpt-3.5-turbo' that supports JSON mode
        # for more reliable structured output.
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert assistant that provides structured data in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # Enable JSON mode
            temperature=0.5, # Lower temperature for more predictable results
        )
        
        # Extract the JSON string from the response
        result_json_str = response.choices[0].message.content
        
        # Parse the JSON string into a Python dictionary
        knowledge = json.loads(result_json_str)

        # Ensure all required keys are in the dictionary
        required_keys = ["fresh_color", "stale_color", "fresh_texture", "stale_texture"]
        if not all(key in knowledge for key in required_keys):
            return f"Error: The model's response was missing required keys for {fruit_name}."

        # Format the final output string according to the template
        formatted_string = (
            f"({knowledge['fresh_color']} or {knowledge['stale_color']}) color, "
            f"({knowledge['fresh_texture']} or {knowledge['stale_texture']}) texture "
            f"{fruit_name}"
        )
        
        return formatted_string

    except json.JSONDecodeError:
        return f"Error: Failed to decode JSON from the model's response for {fruit_name}."
    except Exception as e:
        # Handle potential API errors (e.g., authentication, rate limits)
        return f"An API error occurred for {fruit_name}: {e}"


# --- Example Usage ---
# This block will only run when you execute this script directly.
# It demonstrates how to use the extract_knowledge function.
if __name__ == '__main__':
    fruit1 = "avocado"
    print(f"--- Requesting knowledge for: {fruit1} ---")
    knowledge1 = extract_knowledge(fruit1)
    print(f"Result: {knowledge1}\n")
    # Expected output similar to:
    # Result: (green or dark brownish-black) color, (smooth or firm or wrinkled and mushy) texture avocado

