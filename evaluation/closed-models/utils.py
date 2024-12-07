import requests
import base64
import time
from collections import Counter

def get_prefix_suffix(model):
    if "1.5" in model:
        prefix = '<image>\nUSER: '
        suffix = '\nASSISTANT:'
    elif "1.6" in model and "mistral" in model:
        prefix = '[INST] <image>\n'
        suffix = ' [/INST]'
    elif "1.6" in model and "vicuna" in model:
        prefix = "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and precise answers to the human's questions. USER: <image>\n"
        suffix = "\nASSISTANT:"
    elif "llava-v1.6-34b-hf" in model:
        prefix = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n"
        suffix = "<|im_end|><|im_start|>assistant\n"
    elif "Bunny" in model:
        prefix = "A chat between a curious human and an artificial intelligence assistant.  The assistant gives helpful, detailed, and precise answers to the human's questions. USER: <image>\n"
        suffix = "\nASSISTANT:"
    return prefix, suffix

def get_ordinal(n):
    """Convert an integer into its ordinal representation (e.g., 1 -> 'first')."""
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    if 1 <= n <= len(ordinals):
        return ordinals[n - 1]
    else:
        return f"{n}th"  # Fallback for larger numbers
    
def extract_shapes_from_ground_truth(ground_truth):
    """
    Extract shapes (ignoring colors) from the ground truth.
    Assumes ground truth is a string in the format 'color shape, color shape'.
    """
    shapes = set()
    for item in ground_truth.split(','):
        # Extract the last word (shape) from each item
        shape = item.strip().lower().split()[-1]  # Extract the shape (e.g., 'rectangle')
        shapes.add(shape)
    return shapes

def extract_shape_counts(ground_truth):
    """
    Extract shapes and their counts (ignoring colors) from the ground truth.
    Assumes ground truth is a string in the format 'color shape, color shape'.
    Returns a Counter with shape types as keys and counts as values.
    """
    shape_counts = Counter()
    for item in ground_truth.split(','):
        # Extract the last word (shape) from each item
        shape = item.strip().lower().split()[-1]  # Extract the shape (e.g., 'rectangle')
        shape_counts[shape] += 1
    return shape_counts


class OpenAIModelsAzure:
    def __init__(self, config):
        self.api_key = config["api_key"]
        self.endpoint = config["endpoint"]
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
    
    def encode_image(self, image_path):
        """Encode an image to base64."""    
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('ascii')

    def generate(self, prompt, image_paths):
        """Generate a response using the OpenAI API with text and image data."""
        
        # Encode each image and prepare the message payload
        messages = [{"role": "user", "content": []}]
        
        for i, (text, image_path) in enumerate(zip(prompt, image_paths)):
            # Add text content
            messages[0]["content"].append({"type": "text", "text": text})
            
            # Encode image and add it to the content list
            encoded_image = self.encode_image(image_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })

        print("Messages:\n", messages)

        # Create payload
        payload = {
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800
        }
        # Send the API request with retry logic
        max_retries = 3
        retry_count = 0
        delay = 15  # delay in seconds

        while retry_count < max_retries:
            try:
                response = requests.post(self.endpoint, headers=self.headers, json=payload)
                response.raise_for_status()  # Raise an error for bad responses
                response_data = response.json()
                predicted_answer = response_data["choices"][0]["message"]["content"]
                return predicted_answer
            except requests.RequestException as e:
                print(f"Failed to make the request. Error: {e}. Retrying in {delay} seconds...")
                retry_count += 1
                time.sleep(delay)

        print("Max retries reached. Unable to get a response.")
        return "Error"

        # try:
        #     response = requests.post(self.endpoint, headers=self.headers, json=payload)
        #     response.raise_for_status()  # Raise an error for bad responses
        #     response_data = response.json()
        #     predicted_answer = response_data["choices"][0]["message"]["content"]
        #     return predicted_answer
        # except requests.RequestException as e:
        #     print(f"Failed to make the request. Error: {e}")
        #     return "Error"

# Configuration dictionary
openai_config_gpt4o_azure = {
    "api_key": "",  # Replace with your actual API key
    "endpoint": ""
}

# Example usage
if __name__ == "__main__":
    model = OpenAIModelsAzure(openai_config_gpt4o_azure)
    
    # Define prompts and images
    prompts = [
        "This is the first image. What does it contain?",
        "Now, this is the second image. What about this one?"
    ]
    image_paths = ["../../data/depth_synthetic_2D/images-3-shapes/imgs/img0.jpg", "../../data/depth_synthetic_2D/images-3-shapes/imgs/img1.jpg"]  # Replace with actual paths
    
    # Generate a response
    response = model.generate(prompts, image_paths)
    print(response)





