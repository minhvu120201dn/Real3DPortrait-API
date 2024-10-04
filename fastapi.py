import os
import sys
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import subprocess

from inference.real3d_infer import GeneFace2Infer

# Initialize FastAPI
app = FastAPI()

# Initialize the inference model
class Inferer(GeneFace2Infer):
    def __init__(self, a2m_ckpt, head_ckpt, torso_ckpt, device='cuda'):
        super().__init__(audio2secc_dir=a2m_ckpt, head_model_dir=head_ckpt, torso_model_dir=torso_ckpt, device=device)

    def infer_once_args(self, **inp):
        try:
            out_name = self.infer_once(inp)
            return out_name
        except Exception as e:
            return str(e)

# Initialize model globally
model = Inferer(
    a2m_ckpt='checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt',
    head_ckpt='',
    torso_ckpt='checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt',
    device='cuda:0'
)

# API endpoint for lip-sync inference
@app.post("/lip-sync/")
async def lip_sync(
    src_img: UploadFile = File(...),
    drv_audio: UploadFile = File(None),
    drv_pose: UploadFile = File(None),
    bg_img: UploadFile = File(None),
    blink_mode: str = Form("period"),
    temperature: float = Form(0.2),
    mouth_amp: float = Form(0.45),
    out_mode: str = Form("concat_debug"),
    map_to_init_pose: bool = Form(True),
    low_memory_usage: bool = Form(False),
    hold_eye_opened: bool = Form(False),
    min_face_area_percent: float = Form(0.2)
):
    # Save uploaded files
    src_img_path = f"./{src_img.filename}"
    with open(src_img_path, "wb") as f:
        f.write(await src_img.read())

    drv_audio_path, drv_pose_path, bg_img_path = None, None, None

    if drv_audio:
        drv_audio_path = f"./{drv_audio.filename}"
        with open(drv_audio_path, "wb") as f:
            f.write(await drv_audio.read())
    if drv_pose:
        drv_pose_path = f"./{drv_pose.filename}"
        with open(drv_pose_path, "wb") as f:
            f.write(await drv_pose.read())
    if bg_img:
        bg_img_path = f"./{bg_img.filename}"
        with open(bg_img_path, "wb") as f:
            f.write(await bg_img.read())

    # Prepare input arguments for inference
    inp = {
        "src_image_name": src_img_path,
        "drv_audio_name": drv_audio_path,
        "drv_pose_name": drv_pose_path,
        "bg_image_name": bg_img_path,
        "blink_mode": blink_mode,
        "temperature": temperature,
        "mouth_amp": mouth_amp,
        "out_mode": out_mode,
        "map_to_init_pose": map_to_init_pose,
        "low_memory_usage": low_memory_usage,
        "hold_eye_opened": hold_eye_opened,
        "a2m_ckpt": model.audio2secc_dir,
        "head_ckpt": model.head_model_dir,
        "torso_ckpt": model.torso_model_dir,
        "min_face_area_percent": min_face_area_percent,
    }

    # Call inference
    output_path = model.infer_once_args(**inp)

    # Return the output video
    return {"output_video": output_path}
