from transformers import ViTFeatureExtractor, ViTForImageClassification
import requests
from PIL import Image
import gradio as gr
import warnings
import dotenv
import os
warnings.filterwarnings('ignore')

dotenv.load_dotenv()
# Load the pre-trained Vision Transformer model and feature extractor


model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

api_key = "NslJtHA1MkQVNjmbe0eLrQ==O5nch3BtVgUgim3b"


def identify_image(image_path):
    """Identify the food item in the image."""
    image = Image.open(image_path)
    input = feature_extractor(images=image, return_tensors="pt")
    output = model(**input)
    logits = output.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    food_name = predicted_label.split(',')[0]
    return food_name


def get_calories(food_name):
    """Get the calorie information of the identified food item, excluding unavailable data."""
    api_url = 'https://api.api-ninjas.com/v1/nutrition?query={}'.format(food_name)
    response = requests.get(api_url, headers={'X-Api-Key': api_key})

    if response.status_code == requests.codes.ok:
        raw_nutrition_info = response.json()
        # Filter out fields with the value "Only available for premium subscribers."
        filtered_nutrition_info = []
        for item in raw_nutrition_info:
            filtered_item = {k: v for k, v in item.items() if v != "Only available for premium subscribers."}
            filtered_nutrition_info.append(filtered_item)
        return filtered_nutrition_info
    else:
        return {"Error": response.status_code, "Message": response.text}


def format_nutrition_info(nutrition_info):
    """Format the nutritional information into an HTML table."""
    if "Error" in nutrition_info:
        return f"Error: {nutrition_info['Error']} - {nutrition_info['Message']}"
    
    if len(nutrition_info) == 0:
        return "No nutritional information found."

    nutrition_data = nutrition_info[0]
    table = f"""
    <table border="1" style="width: 100%; border-collapse: collapse;">
        <tr><th colspan="2" style="text-align: center;"><b>Nutrition Facts</b></th></tr>
        <tr><td colspan="2" style="text-align: center;"><b>Food Name: {nutrition_data['name']}</b></td></tr>
        <tr>
            <td style="text-align: left;"><b>Total Fat (g)</b></td><td style="text-align: right;">{nutrition_data['fat_total_g']}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Saturated Fat (g)</b></td><td style="text-align: right;">{nutrition_data['fat_saturated_g']}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Sodium (mg)</b></td><td style="text-align: right;">{nutrition_data['sodium_mg']}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Potassium (mg)</b></td><td style="text-align: right;">{nutrition_data['potassium_mg']}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Cholesterol (mg)</b></td><td style="text-align: right;">{nutrition_data['cholesterol_mg']}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Total Carbohydrates (g)</b></td><td style="text-align: right;">{nutrition_data['carbohydrates_total_g']}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Fiber (g)</b></td><td style="text-align: right;">{nutrition_data['fiber_g']}</td>
        </tr>
        <tr>
            <td style="text-align: left;"><b>Sugar (g)</b></td><td style="text-align: right;">{nutrition_data['sugar_g']}</td>
        </tr>
    </table>
    """
    return table

def main_process(image_path):
    """Identify the food item and fetch its calorie information."""
    food_name = identify_image(image_path)
    nutrition_info = get_calories(food_name)
    formatted_nutrition_info = format_nutrition_info(nutrition_info)
    return formatted_nutrition_info

# Define the Gradio interface
def gradio_interface(image):
    formatted_nutrition_info = main_process(image)
    return formatted_nutrition_info

# Create the Gradio UI
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="filepath"),
    outputs="html",
    title="Food Identification and Nutrition Info",
    description="Upload an image of food to get nutritional information.",
    allow_flagging="never"  # Disable flagging
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()