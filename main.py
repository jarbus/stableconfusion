import io
import logging as log
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

import torch
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
# this solver reduces time needed per generation
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

pipeline = pipeline.to("cuda")

app = FastAPI()
@app.get("/")
def submit_prompt(prompt: str):
    generator = torch.Generator("cuda").manual_seed(0)
    print(f"Generating image for prompt: {prompt}")
    image = pipeline(prompt, 
                     generator=generator,
                     num_inference_steps=20
                     ).images[0]
    image_stream = io.BytesIO()
    image.save(image_stream, format="JPEG")
    image_stream.seek(0)
    # response = StreamingResponse(image_stream, media_type="image/jpeg")
    print(f"Returning photo for prompt: {prompt}")
    image_bytes = image_stream.read()
    image_b64 = base64.b64encode(image_bytes)

    return image_b64
