import random
import uuid
import time
import os
import shutil
import glob
import traceback
from workflow import WanVideoInfiniteTalk
import runpod
from rp_utils import (
    process_input_image,
    process_input_audio,
    upload_to_supabase,
    cleanup_temp_files,
    get_file_size_mb
)

# Initialize the workflow once at startup
print('clone successful')
print("Initializing WanVideoMultiTalk workflow...")
init_start_time = time.time()
workflow = WanVideoInfiniteTalk()
init_end_time = time.time()
init_time = init_end_time - init_start_time
print(f"Workflow initialization time: {init_time:.2f} seconds")

def handler(event):
    """
    RunPod handler for WanVideoMultiTalk
    
    Expected input parameters:
    - request_id (optional): Unique request identifier, defaults to random UUID
    - image (required): Input image as base64 string or URL
    - audio (required): Input audio as URL
    - lora_path (optional): Path to custom LoRA file (e.g., "/workspace/custom_lora.safetensors")
    - positive_prompt (optional): Positive prompt for video generation
    - negative_prompt (optional): Negative prompt for video generation
    - seed (optional): Random seed for reproducible results
    """
    temp_files = []  # Track temporary files for cleanup
    
    try:
        # Extract input parameters
        input_data = event.get("input", {})
        
        # Generate or use provided request_id
        request_id = input_data.get("request_id")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Validate required parameters
        if "image" not in input_data:
            return {
                "request_id": request_id,
                "error": "Missing required parameter: image"
            }
        
        if "audio" not in input_data:
            return {
                "request_id": request_id,
                "error": "Missing required parameter: audio"
            }
        
        # Extract parameters with defaults
        image_input = input_data["image"]
        audio_input = input_data["audio"]
        audio_duration_limit = input_data.get("audio_limit", 120)
        lora_path = input_data.get("lora_path", None)  # Optional LoRA path
        fps = input_data.get("fps", 25)
        
        # User-configurable parameters
        positive_prompt = input_data.get("positive_prompt", 
            "A woman speakinng passionately about a face cream that she loves")
        negative_prompt = input_data.get("negative_prompt", 
            "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
        seed = input_data.get("seed",42)  # Optional seed
        
        # Use provided seed or generate random one
        if seed is None:
            seed = random.randint(1, 2**64)
        
        # Validate that all text prompts are strings
        text_params = {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt
        }
        
        for param_name, param_value in text_params.items():
            if not isinstance(param_value, str):
                return {
                    "request_id": request_id,
                    "error": f"{param_name} must be a string"
                }
        
        # Validate seed parameter (only user-configurable numeric parameter)
        if seed is not None and not isinstance(seed, (int, float)):
            return {
                "request_id": request_id,
                "error": "seed must be a number"
            }
        
        # Process LoRA path if provided
        processed_lora_path = None
        if lora_path is not None:
            if not isinstance(lora_path, str):
                return {
                    "request_id": request_id,
                    "error": "lora_path must be a string"
                }
            
            # Replace /workspace with /runpod-volume if present
            if lora_path.startswith("/workspace/"):
                lora_path = lora_path.replace("/workspace", "/runpod-volume", 1)
            
            # Verify the file exists at the runpod-volume path
            if not os.path.exists(lora_path):
                return {
                    "request_id": request_id,
                    "error": f"LoRA file not found at path: {lora_path}"
                }
            
            # Extract the path relative to /workspace or /runpod-volume
            if lora_path.startswith("/workspace/"):
                lora_filename = lora_path[10:]  # Remove "/workspace/"
            elif lora_path.startswith("/runpod-volume/"):
                lora_filename = lora_path[15:]  # Remove "/runpod-volume/"
            else:
                lora_filename = os.path.basename(lora_path)  # Fallback to just filename
            comfyui_lora_path = f"/app/ComfyUI/models/loras/{lora_filename}"
            
            # Check if LoRA already exists in ComfyUI directory
            if not os.path.exists(comfyui_lora_path):
                # Create ComfyUI loras directory and any subdirectories if they don't exist
                os.makedirs(os.path.dirname(comfyui_lora_path), exist_ok=True)
                
                try:
                    # Try to create a symlink first (more efficient)
                    os.symlink(lora_path, comfyui_lora_path)
                    print(f"Created symlink: {lora_path} -> {comfyui_lora_path}")
                except OSError:
                    # If symlink fails (e.g., cross-filesystem), copy the file
                    shutil.copy2(lora_path, comfyui_lora_path)
                    print(f"Copied LoRA file: {lora_path} -> {comfyui_lora_path}")
            
            processed_lora_path = lora_filename
            print(f"LoRA path processed: {lora_filename}")
        
        # Clear ComfyUI input and output folders at the start of each request
        shutil.rmtree("ComfyUI/input", ignore_errors=True)
        shutil.rmtree("ComfyUI/output", ignore_errors=True)
        
        # Process input files
        print(f"Processing input files for request_id: {request_id}")
        input_start_time = time.time()
        image_path = process_input_image(image_input, request_id)
        # Note: image_path is not added to temp_files as it's saved to ComfyUI/input/input_image.png
        
        audio_path = process_input_audio(audio_input, request_id, audio_duration_limit)
        # Note: audio_path is not added to temp_files as it's saved to ComfyUI/input/input_audio.mp3
        input_end_time = time.time()
        input_processing_time = input_end_time - input_start_time
        print(f"Input processing time: {input_processing_time:.2f} seconds")
        
        # Log request details before starting generation
        payload = {
            "request_id": request_id,
            "image_type": "base64" if image_input.startswith(('data:', '/9j/', 'iVBORw0KGgo')) else "url",
            "audio_type": "url",
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "lora_path": processed_lora_path,
            "fps": fps
        }
        print(f"Starting video generation for request_id: {request_id}")
        print(f"Payload: {payload}")
        
        # Run the workflow
        print(f"Starting video generation for request_id: {request_id}")
        inference_start_time = time.time()
        result = workflow(
            image=image_path,
            audio=audio_path,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            lora_path=processed_lora_path,
            fps=fps
        )
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        
        print(f"Workflow completed for request_id: {request_id}")
        print(f"Result type: {type(result)}")
        
        # The workflow returns a video combine result, we need to extract the video file
        # Look for the output video file in ComfyUI/output/ directory
        
        if hasattr(result, 'filename') and result.filename:
            video_path = result.filename
        elif isinstance(result, str) and os.path.exists(result):
            video_path = result
        else:
            # Try to find the generated video file in ComfyUI/output/
            output_dir = "ComfyUI/output"
            video_files = glob.glob(os.path.join(output_dir, "*.mp4"))
            if video_files:
                # Get the most recently created file
                video_path = max(video_files, key=os.path.getctime)
            else:
                raise Exception(f"Could not locate generated video file in {output_dir}")
        
        print(f"Found video file: {video_path}")
        
        # Generate unique filename for upload
        file_extension = os.path.splitext(video_path)[1]
        upload_filename = f"{request_id}_output{file_extension}"
        
        # Upload to Supabase
        print(f"Uploading video to Supabase bucket")
        upload_start_time = time.time()
        video_url = upload_to_supabase(video_path, upload_filename)
        upload_end_time = time.time()
        upload_time = upload_end_time - upload_start_time
        
        # Get file size for logging
        file_size_mb = get_file_size_mb(video_path)
        print(f"Video file size: {file_size_mb:.2f} MB")
        print(f"Upload time: {upload_time:.2f} seconds")
        print(f"Video uploaded successfully: {video_url}")
        
        # Clean up temporary files
        cleanup_temp_files(temp_files)
        
        # Log completion before sending result
        print(f"Video generation and upload completed for request_id: {request_id}")
        
        # Calculate total processing time
        total_time = input_processing_time + inference_time + upload_time
        
        return {
            "request_id": request_id,
            "success": True,
            "video": video_url,
            "fps": fps
        }
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error processing request {request_id if 'request_id' in locals() else 'unknown'}: {str(e)}")
        print(f"Full traceback for request {request_id if 'request_id' in locals() else 'unknown'}: \n{error_traceback}")
        
        # Clean up temporary files on error
        cleanup_temp_files(temp_files)
        
        return {
            "request_id": request_id if 'request_id' in locals() else str(uuid.uuid4()),
            "error": f"Processing failed: {str(e)}"
        }

# Example usage for testing
if __name__ == "__main__":
    print("Running serverless WanVideo generation service")
    runpod.serverless.start({"handler": handler})
