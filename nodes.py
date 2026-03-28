import os
import torch
import numpy as np
import folder_paths
import unicodedata
from PIL import Image
# 修正匯入：使用官方認證的 StableDiffusion Pipeline
from optimum.intel.openvino import OVStableDiffusionPipeline
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# =========================================================
# 🧠 1. Florence-2 Vision Director
# =========================================================
class Florence2VisionDirector:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_id = ""
        self.device = "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",), 
                "pose_image": ("IMAGE",),
                "character_image": ("IMAGE",),
                "model_id": (["microsoft/Florence-2-base", "microsoft/Florence-2-large"], {"default": "microsoft/Florence-2-base"}),
                "task": (["<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"], {"default": "<DETAILED_CAPTION>"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("pos_cond", "neg_cond", "raw_text")
    FUNCTION = "analyze"
    CATEGORY = "Z-Image-Turbo-Lab"

    def analyze(self, clip, pose_image, character_image, model_id, task, **kwargs):
        # 確保模型載入 (針對 16GB RAM 優化)
        if self.model is None or self.current_model_id != model_id:
            print(f"[Z-Turbo] 載入 Florence-2: {model_id}")
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            config._attn_implementation = "eager"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.current_model_id = model_id

        def get_single_desc(img_tensor, m, p):
            pil_img = Image.fromarray(np.clip(255. * img_tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")
            inputs = p(text=task, images=pil_img, return_tensors="pt").to(self.device)
            generated_ids = m.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=128)
            results = p.batch_decode(generated_ids, skip_special_tokens=True)[0]
            parsed = p.post_process_generation(results, task=task, image_size=pil_img.size)
            res_text = parsed[task] if task in parsed else str(parsed)
            return unicodedata.normalize('NFKC', str(res_text))

        try:
            desc_char = get_single_desc(character_image, self.model, self.processor)
            desc_pose = get_single_desc(pose_image, self.model, self.processor)
            pos_prompt = f"{desc_char}, posing as {desc_pose}"
            print(f"[Z-Turbo] 產出的 Prompt: {pos_prompt}")
        except Exception as e:
            print(f"[Z-Turbo] 推論錯誤: {e}")
            pos_prompt = "a person, high quality"

        tokens = clip.tokenize(pos_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "raw_text": pos_prompt}]], [], pos_prompt)

# =========================================================
# ⚡ 2. Z-Image Turbo Dual-Prompt (OpenVINO 優化版)
# =========================================================
class ZImageTurboOpenVINO:
    def __init__(self):
        self.pipe = None
        self.current_model = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "Z_image_turbo_OPENVINO"}),
                "positive_cond": ("CONDITIONING",),
                "negative_cond": ("CONDITIONING",),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "steps": ("INT", {"default": 4}),
                "cfg": ("FLOAT", {"default": 1.0}),
                "seed": ("INT", {"default": 0}),
                "device": (["GPU", "CPU"], {"default": "GPU"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image-Turbo-Lab"

    def generate(self, model_name, positive_cond, negative_cond, width, height, steps, cfg, seed, device):
        # 提取文字
        pos_text = positive_cond[0][1].get("raw_text", "a high quality image") if isinstance(positive_cond, list) else ""
        
        # 載入 OpenVINO Pipeline (修正為官方標準)
        if self.pipe is None or self.current_model != model_name:
            load_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
            if not os.path.exists(load_path):
                load_path = os.path.join(folder_paths.models_dir, model_name)
            
            print(f"[Z-Turbo] 載入 OpenVINO 模型: {load_path}")
            # 使用官方 OVStableDiffusionPipeline，確保硬體加速正常
            self.pipe = OVStableDiffusionPipeline.from_pretrained(
                load_path, 
                device=device,
                compile=True # 針對 Iris Xe 內顯編譯
            )
            self.current_model = model_name

        generator = torch.Generator("cpu").manual_seed(seed)
        result = self.pipe(
            prompt=pos_text, 
            width=width, 
            height=height, 
            num_inference_steps=steps, 
            guidance_scale=cfg, 
            generator=generator
        ).images[0]
        
        img = np.array(result).astype(np.float32) / 255.0
        return (torch.from_numpy(img)[None,],)

# =========================================================
# 🚀 註冊節點
# =========================================================
NODE_CLASS_MAPPINGS = {
    "Florence2Director": Florence2VisionDirector,
    "ZImageTurboOV": ZImageTurboOpenVINO
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Florence2Director": "🧠 Florence-2 Vision Director",
    "ZImageTurboOV": "⚡ Z-Image Turbo Dual-Prompt (OV)"
}