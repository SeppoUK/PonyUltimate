PonyUltimatePro is an all-in-one generation node customized specifically for Pony Diffusion workflows. It streamlines the process by automatically handling Pony's mandatory scoring tags, offering instant aesthetic style presets, and integrating automated face detailing into a single, easy-to-use node.

✨ Key Features
Auto-Scoring Tags: Automatically prepends score_9, score_8_up, score_7_up to your positive prompt. It also automatically adds score_5_up, score_4_up to your negative prompt.

Aesthetic Styles: Includes a built-in dropdown to apply instant style modifiers like Cinematic, Photographic, Digital Art, Anime, and Vintage.

SDXL Resolutions: Features a dropdown for optimized aspect ratios, including 1024x1024 (1:1), 832x1216 (Portrait), 1216x832 (Landscape), 768x1344 (9:16), and 1344x768 (16:9).

Integrated Face Fix: Uses the FaceDetailer and Ultralytics YOLO models to automatically detect and repair faces in your generation.

Comparison Mode: Offers a show_comparison toggle to output a side-by-side view of the base image alongside the face-detailed image.

📥 Inputs
ckpt_name: Select your checkpoint model. (Defaults to a Pony model folder path).

style: Choose an aesthetic style or set it to "None".

aspect_ratio: Select your desired image dimensions from the preset list.

positive / negative: Your main text prompts. (You do not need to type the score tags manually, the node does it for you!).

seed, steps, cfg, sampler_name, scheduler: Standard generation settings.

face_detailer: Enable or disable the automated face fix.

face_denoise: Control how strongly the face detailer alters the original face.

show_comparison: Toggle "Yes" to output a side-by-side before and after image.

📤 Outputs
IMAGE: The final generated image, or the side-by-side comparison if that setting is enabled.

MODEL: The loaded checkpoint model, ready to be passed to other nodes.

VAE: The loaded VAE, ready to be passed to other nodes.

⚠️ Requirements
Because this node utilizes advanced face detection and detailing under the hood, it relies on external classes to function. You must have the following installed in your ComfyUI environment:

ComfyUI-Impact-Pack: This node specifically calls the FaceDetailer and UltralyticsDetectorProvider classes provided by the Impact Pack. It requires the face_yolov8m.pt bounding box model to detect faces.
