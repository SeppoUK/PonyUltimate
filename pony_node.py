import torch
import folder_paths
import comfy.sd
import comfy.utils
import nodes
import model_management

class PonyUltimatePro:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "Pony\\realismByStableYogi_v60FP16.safetensors"}),
                "style": (["None", "Cinematic", "Photographic", "Digital Art", "Anime", "Vintage"], {"default": "Photographic"}),
                "aspect_ratio": (["1024x1024 (1:1)", "832x1216 (Portrait)", "1216x832 (Landscape)", "768x1344 (9:16)", "1344x768 (16:9)"], {"default": "832x1216 (Portrait)"}),
                "positive": ("STRING", {"default": "1girl, solo, masterpiece", "multiline": True}),
                "negative": ("STRING", {"default": "source_furry, source_pony", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m_sde"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "face_detailer": (["Enabled", "Disabled"], {"default": "Enabled"}),
                "face_denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "show_comparison": (["Yes", "No"], {"default": "No"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "VAE")
    FUNCTION = "gen"
    CATEGORY = "PonyUltimate"

    def gen(self, ckpt_name, style, aspect_ratio, positive, negative, seed, steps, cfg, sampler_name, scheduler, face_detailer, face_denoise, show_comparison):
        print(f"🚀 Starting Pony Pro Generation [Seed: {seed}]")
        
        style_prompts = {
            "Cinematic": "cinematic lighting, dramatic shadows, 8k, film grain",
            "Photographic": "raw photo, realistic skin texture, bokeh, 35mm lens",
            "Digital Art": "vibrant colors, sharp lines, concept art",
            "Anime": "anime style, cel shaded, vibrant",
            "Vintage": "1990s film, grainy, faded colors"
        }
        full_positive = f"score_9, score_8_up, score_7_up, {positive}, {style_prompts.get(style, '')}"
        full_negative = f"score_5_up, score_4_up, {negative}"

        res_map = {"1024x1024 (1:1)": (1024, 1024), "832x1216 (Portrait)": (832, 1216), "1216x832 (Landscape)": (1216, 832), "768x1344 (9:16)": (768, 1344), "1344x768 (16:9)": (1344, 768)}
        w, h = res_map[aspect_ratio]
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        model, clip, vae = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True)[:3]

        def encode(clip_obj, text):
            tokens = clip_obj.tokenize(text)
            cond, pooled = clip_obj.encode_from_tokens(tokens, return_pooled=True)
            return [[cond, {"pooled_output": pooled}]]

        print(f"🎨 Encoding Prompts...")
        pc, nc = encode(clip, full_positive), encode(clip, full_negative)
        
        print(f"🎨 Rendering Base Image...")
        latent = {"samples": torch.zeros([1, 4, h // 8, w // 8])}
        samples = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, pc, nc, latent, denoise=1.0)[0]
        base_image = nodes.VAEDecode().decode(vae, samples)[0]

        final_output = base_image

        if face_detailer == "Enabled":
            print("🔍 Face Detailer: Initializing...")
            model_management.soft_empty_cache()
            
            DetectorClass = nodes.NODE_CLASS_MAPPINGS.get("UltralyticsDetectorProvider")
            DetailerClass = nodes.NODE_CLASS_MAPPINGS.get("FaceDetailer")
            
            if DetectorClass and DetailerClass:
                try:
                    det_inst = DetectorClass()
                    bbox_detector = None
                    for model_path in ["bbox/face_yolov8m.pt", "face_yolov8m.pt"]:
                        for method in ["load_v1", "load_model", "doit"]:
                            if hasattr(det_inst, method):
                                try:
                                    res = getattr(det_inst, method)(model_name=model_path)
                                    bbox_detector = res[0] if isinstance(res, tuple) else res
                                    if bbox_detector: break
                                except: continue
                        if bbox_detector: break
                    
                    if bbox_detector:
                        print("💎 Face Detailer: Detector loaded. Starting full pass...")
                        detailer_inst = DetailerClass()
                        
                        # Added missing: positive, negative, and bbox_threshold
                        detailed_image = detailer_inst.doit(
                            image=base_image, 
                            model=model, clip=clip, vae=vae,
                            guide_size=768, guide_size_for=True, max_size=1024,
                            seed=seed, steps=20, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                            denoise=face_denoise, feather=5, noise_mask=True, force_inpaint=True,
                            wildcard="", cycle=1,
                            bbox_dilation=10, bbox_crop_factor=3.0,
                            sam_detection_hint="center-1", sam_dilation=0, sam_threshold=0.93,
                            sam_bbox_expansion=0, sam_mask_hint_threshold=0.7, sam_mask_hint_use_negative=False,
                            drop_size=10, bbox_detector=bbox_detector,
                            positive=pc, negative=nc, bbox_threshold=0.5 # The final missing links
                        )[0]
                        
                        if show_comparison == "Yes":
                            final_output = torch.cat((base_image, detailed_image), dim=2)
                        else:
                            final_output = detailed_image
                except Exception as e:
                    print(f"❌ Detailer Error: {e}")

        print("✅ Generation Complete.")
        return (final_output, model, vae)