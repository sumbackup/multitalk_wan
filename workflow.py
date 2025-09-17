import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    try:
        # Try async version first
        loop.run_until_complete(init_extra_nodes())
    except TypeError:
        # Fallback to sync version if it's not async
        init_extra_nodes()


from nodes import LoadImage, NODE_CLASS_MAPPINGS, CLIPVisionLoader


class WanVideoMultiTalk:
    def __init__(self):
        import_custom_nodes()
        
        # Initialize all model loaders and nodes
        self.multitalkmodelloader = NODE_CLASS_MAPPINGS["MultiTalkModelLoader"]()
        self.wanvideovaeloader = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
        self.loadimage = LoadImage()
        self.wanvideoblockswap = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
        self.loadwanvideot5textencoder = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
        self.downloadandloadwav2vecmodel = NODE_CLASS_MAPPINGS["DownloadAndLoadWav2VecModel"]()
        self.clipvisionloader = CLIPVisionLoader()
        self.wanvideotorchcompilesettings = NODE_CLASS_MAPPINGS["WanVideoTorchCompileSettings"]()
        self.vhs_loadaudioupload = NODE_CLASS_MAPPINGS["VHS_LoadAudioUpload"]()
        self.wanvideoloraselect = NODE_CLASS_MAPPINGS["WanVideoLoraSelect"]()
        self.wanvideomodelloader = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
        self.vram_debug = NODE_CLASS_MAPPINGS["VRAM_Debug"]()
        self.get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
        self.imageresizekjv2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
        self.easy_mathfloat = NODE_CLASS_MAPPINGS["easy mathFloat"]()
        self.cr_float_to_integer = NODE_CLASS_MAPPINGS["CR Float To Integer"]()
        self.wanvideoclipvisionencode = NODE_CLASS_MAPPINGS["WanVideoClipVisionEncode"]()
        self.wanvideoimagetovideoencode = NODE_CLASS_MAPPINGS["WanVideoImageToVideoEncode"]()
        self.wanvideotextencode = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
        self.audiocrop = NODE_CLASS_MAPPINGS["AudioCrop"]()
        self.audioseparation = NODE_CLASS_MAPPINGS["AudioSeparation"]()
        self.layermask_segmentanythingultra_v2 = NODE_CLASS_MAPPINGS["LayerMask: SegmentAnythingUltra V2"]()
        self.multitalkwav2vecembeds = NODE_CLASS_MAPPINGS["MultiTalkWav2VecEmbeds"]()
        self.wanvideosampler = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
        self.wanvideodecode = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
        self.vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        # Load models
        self.multitalkmodelloader_120 = self.multitalkmodelloader.loadmodel(
            model="WanVideo_2_1_Multitalk_14B_fp8_e4m3fn.safetensors"
        )

        self.wanvideovaeloader_129 = self.wanvideovaeloader.loadmodel(
            model_name="Wan2_1_VAE_bf16.safetensors", precision="bf16"
        )

        self.wanvideoblockswap_134 = self.wanvideoblockswap.setargs(
            blocks_to_swap=15,
            offload_img_emb=False,
            offload_txt_emb=False,
            use_non_blocking=True,
            vace_blocks_to_swap=0,
            prefetch_blocks=0,
            block_swap_debug=False,
        )

        self.loadwanvideot5textencoder_136 = self.loadwanvideot5textencoder.loadmodel(
            model_name="umt5_xxl_fp16.safetensors",
            precision="bf16",
            load_device="offload_device",
            quantization="disabled",
        )

        self.downloadandloadwav2vecmodel_137 = self.downloadandloadwav2vecmodel.loadmodel(
            model="TencentGameMate/chinese-wav2vec2-base",
            base_precision="fp16",
            load_device="main_device",
        )

        self.clipvisionloader_173 = self.clipvisionloader.load_clip(
            clip_name="clip_vision_h.safetensors"
        )

        self.wanvideotorchcompilesettings_177 = self.wanvideotorchcompilesettings.set_args(
            backend="inductor",
            fullgraph=False,
            mode="default",
            dynamic=False,
            dynamo_cache_size_limit=64,
            compile_transformer_blocks_only=True,
            dynamo_recompile_limit=128,
        )

        self.wanvideoloraselect_238 = self.wanvideoloraselect.getlorapath(
            lora="FusionX_FaceNaturalizer.safetensors",
            strength=1,
            low_mem_load=False,
            merge_loras=True,
            unique_id=11824802763977056194,
        )

        # Additional LoRA selections
        self.wanvideoloraselect_211 = self.wanvideoloraselect.getlorapath(
            lora="Wan14Bi2vFusioniX_pure_fp16.safetensors",
            strength=1,
            low_mem_load=False,
            merge_loras=True,
            prev_lora=get_value_at_index(self.wanvideoloraselect_238, 0),
            unique_id=10030556395511193686,
        )

        self.wanvideoloraselect_138 = self.wanvideoloraselect.getlorapath(
            lora="Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors",
            strength=0.8000000000000002,
            low_mem_load=False,
            merge_loras=True,
            prev_lora=get_value_at_index(self.wanvideoloraselect_211, 0),
            unique_id=6129929711643136840,
        )

    @torch.inference_mode()
    def __call__(self, *args, **kwargs):
        # Get parameters from kwargs
        image_path = kwargs.get("image", "input_image.png")
        audio_path = kwargs.get("audio", "input_audio.mp3")
        positive_prompt = kwargs.get("positive_prompt", "A woman speakinng passionately about a face cream that she loves")
        negative_prompt = kwargs.get("negative_prompt", "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
        steps = kwargs.get("steps", 4)
        cfg = kwargs.get("cfg", 1.0)
        shift = kwargs.get("shift", 11.0)
        seed = kwargs.get("seed", random.randint(1, 2**64))
        lora_path = kwargs.get("lora_path", None)
        filename_prefix = kwargs.get("filename_prefix", "WanVideo2_1_multitalk")
        fps = kwargs.get("fps", 25)
        # Model loading
        if lora_path is not None:
            final_lora = self.wanvideoloraselect.getlorapath(
                lora=lora_path,
                strength=1,
                low_mem_load=False,
                merge_loras=True,
                prev_lora=get_value_at_index(self.wanvideoloraselect_138, 0),
                unique_id=87987,
            )
            print('loaded lora successfully')
        else:
            print('no custom lora to load')
            final_lora = self.wanvideoloraselect_138

        wanvideomodelloader_122 = self.wanvideomodelloader.loadmodel(
            model="wan2.1_i2v_720p_14B_bf16.safetensors",
            base_precision="fp16_fast",
            quantization="fp8_e4m3fn",
            load_device="offload_device",
            attention_mode="sdpa",
            compile_args=get_value_at_index(self.wanvideotorchcompilesettings_177, 0),
            block_swap_args=get_value_at_index(self.wanvideoblockswap_134, 0),
            lora=get_value_at_index(final_lora, 0),
            multitalk_model=get_value_at_index(self.multitalkmodelloader_120, 0),
        )

        # VRAM debug
        vram_debug_206 = self.vram_debug.VRAMdebug(
            empty_cache=True,
            gc_collect=True,
            unload_all_models=False,
            any_input=get_value_at_index(wanvideomodelloader_122, 0),
        )
        # Load input image and audio
        loadimage_133 = self.loadimage.load_image(image=image_path)
        vhs_loadaudioupload_212 = self.vhs_loadaudioupload.load_audio(
            audio=audio_path, start_time=0, duration=0
        )


        get_image_size_239 = self.get_image_size.get_size(
            image=get_value_at_index(loadimage_133, 0)
        )

        imageresizekjv2_225 = self.imageresizekjv2.resize(
            width=get_value_at_index(get_image_size_239, 0),
            height=get_value_at_index(get_image_size_239, 1),
            upscale_method="nearest-exact",
            keep_proportion="resize",
            pad_color="0, 0, 0",
            crop_position="center",
            divisible_by=16,
            device="cpu",
            image=get_value_at_index(loadimage_133, 0),
            unique_id=5063178432147508168,
        )

        easy_mathfloat_214 = self.easy_mathfloat.float_math_operation(
            a=get_value_at_index(vhs_loadaudioupload_212, 1),
            b=16.000000000000004,
            operation="multiply",
        )

        cr_float_to_integer_217 = self.cr_float_to_integer.convert(
            _float=get_value_at_index(easy_mathfloat_214, 0)
        )

        wanvideoclipvisionencode_193 = self.wanvideoclipvisionencode.process(
            strength_1=1,
            strength_2=1,
            crop="center",
            combine_embeds="average",
            force_offload=True,
            tiles=0,
            ratio=0.5000000000000001,
            clip_vision=get_value_at_index(self.clipvisionloader_173, 0),
            image_1=get_value_at_index(imageresizekjv2_225, 0),
        )

        wanvideoimagetovideoencode_207 = self.wanvideoimagetovideoencode.process(
            width=get_value_at_index(imageresizekjv2_225, 1),
            height=get_value_at_index(imageresizekjv2_225, 2),
            num_frames=get_value_at_index(cr_float_to_integer_217, 0),
            noise_aug_strength=0,
            start_latent_strength=1,
            end_latent_strength=1,
            force_offload=True,
            fun_or_fl2v_model=False,
            tiled_vae=False,
            vae=get_value_at_index(self.wanvideovaeloader_129, 0),
            clip_embeds=get_value_at_index(wanvideoclipvisionencode_193, 0),
            start_image=get_value_at_index(imageresizekjv2_225, 0),
        )

        wanvideotextencode_135 = self.wanvideotextencode.process(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            force_offload=True,
            use_disk_cache=False,
            device="gpu",
            t5=get_value_at_index(self.loadwanvideot5textencoder_136, 0),
        )

        audiocrop_159 = self.audiocrop.main(
            start_time="0:00",
            end_time="2:00",
            audio=get_value_at_index(vhs_loadaudioupload_212, 0),
        )

        audioseparation_170 = self.audioseparation.main(
            chunk_fade_shape="linear",
            chunk_length=10,
            chunk_overlap=0.1,
            audio=get_value_at_index(audiocrop_159, 0),
        )

        layermask_segmentanythingultra_v2_200 = (
            self.layermask_segmentanythingultra_v2.segment_anything_ultra_v2(
                sam_model="sam_vit_h (2.56GB)",
                grounding_dino_model="GroundingDINO_SwinT_OGC (694MB)",
                threshold=0.3,
                detail_method="VITMatte",
                detail_erode=6,
                detail_dilate=6,
                black_point=0.15,
                white_point=0.99,
                process_detail=True,
                prompt="character",
                device="cuda",
                max_megapixels=2,
                cache_model=False,
                image=get_value_at_index(imageresizekjv2_225, 0),
            )
        )

        multitalkwav2vecembeds_194 = self.multitalkwav2vecembeds.process(
            normalize_loudness=True,
            num_frames=get_value_at_index(cr_float_to_integer_217, 0),
            fps=fps,
            audio_scale=1,
            audio_cfg_scale=1,
            multi_audio_type="para",
            wav2vec_model=get_value_at_index(self.downloadandloadwav2vecmodel_137, 0),
            audio_1=get_value_at_index(audioseparation_170, 3),
            ref_target_masks=get_value_at_index(
                layermask_segmentanythingultra_v2_200, 1
            ),
        )

        wanvideosampler_128 = self.wanvideosampler.process(
            steps=steps,
            cfg=cfg,
            shift=shift,
            seed=seed,
            force_offload=True,
            scheduler="flowmatch_distill",
            riflex_freq_index=0,
            denoise_strength=1,
            batched_cfg=False,
            rope_function="comfy",
            start_step=0,
            end_step=-1,
            add_noise_to_samples=False,
            model=get_value_at_index(vram_debug_206, 0),
            image_embeds=get_value_at_index(wanvideoimagetovideoencode_207, 0),
            text_embeds=get_value_at_index(wanvideotextencode_135, 0),
            multitalk_embeds=get_value_at_index(multitalkwav2vecembeds_194, 0),
        )

        wanvideodecode_130 = self.wanvideodecode.decode(
            enable_vae_tiling=False,
            tile_x=272,
            tile_y=272,
            tile_stride_x=144,
            tile_stride_y=128,
            normalization="default",
            vae=get_value_at_index(self.wanvideovaeloader_129, 0),
            samples=get_value_at_index(wanvideosampler_128, 0),
        )

        vhs_videocombine_131 = self.vhs_videocombine.combine_video(
            frame_rate=fps,
            loop_count=0,
            filename_prefix=filename_prefix,
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=19,
            save_metadata=True,
            trim_to_audio=False,
            pingpong=False,
            save_output=True,
            images=get_value_at_index(wanvideodecode_130, 0),
            audio=get_value_at_index(audiocrop_159, 0),
            unique_id=10543100069109904756,
        )

        # Cleanup: Free GPU memory
        del wanvideomodelloader_122
        del vram_debug_206
        torch.cuda.empty_cache()

        return vhs_videocombine_131


if __name__ == "__main__":
    model = WanVideoMultiTalk()
    result = model(
        positive_prompt="A woman speakinng passionately about a face cream that she loves",
        negative_prompt="bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        steps=4,
        cfg=1.0000000000000002,
        shift=11.000000000000002,
        seed=random.randint(1, 2**64),
        num_iterations=10,
        filename_prefix="WanVideo2_1_multitalk"
    )
    print('result:', result)
