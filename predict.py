import os
import json
import mimetypes
import shutil
import re
import requests
import tarfile
import tempfile
from typing import Any, List

from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper
from huggingface_hub import HfApi

# Directories for inputs/outputs
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/quicktime", ".mov")

# Use the updated JSON file:
api_json_file = "t2v-lora-updated.json"

# Ensure HF Hub is online for LoRA downloads
if "HF_HUB_OFFLINE" in os.environ:
    del os.environ["HF_HUB_OFFLINE"]


class Predictor(BasePredictor):
    def setup(self):
        """
        Start ComfyUI, ensuring it doesn't attempt to download our local LoRA file
        before running. We do this by blanking out node 41's "lora" field so the
        weight downloader never sees it.
        """
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # 1. Load the main workflow JSON
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # 2. Blank node 41's "lora" so ComfyUI won't attempt to download it
        if workflow.get("41") and "lora_name" in workflow["41"]["inputs"]:
            workflow["41"]["inputs"]["lora_name"] = ""

        # 3. Only handle the base model weights here
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "hunyuan_video_720_fp8_e4m3fn.safetensors",
                "hunyuan_video_vae_bf16.safetensors",
                "clip-vit-large-patch14",
                "llava-llama-3-8b-text-encoder-tokenizer",
            ],
        )

    def copy_lora_file(self, lora_url: str) -> str:
        """
        Download the user-provided LoRA file from either:
          1) A direct URL to a .safetensors file (http/https).
          2) A Hugging Face repo ID (e.g. "username/repo"), automatically finding
             the first .safetensors file and using its main branch URL.

        The file is placed into ComfyUI/models/loras/ and ensured to be a
        '.safetensors' file. Returns the final filename.
        """
        # Create/ensure our target folder exists
        lora_dir = os.path.join("ComfyUI", "models", "loras")
        os.makedirs(lora_dir, exist_ok=True)

        # If this looks like an http(s) link, handle direct download
        if re.match(r"^https?:\/\/", lora_url):
            # Attempt to derive a local filename from the URL
            filename = os.path.basename(lora_url)
            if not filename.lower().endswith(".safetensors"):
                filename += ".safetensors"

            dst_path = os.path.join(lora_dir, filename)

            # Download and write to the local destination
            resp = requests.get(lora_url)
            resp.raise_for_status()
            with open(dst_path, "wb") as f:
                f.write(resp.content)

            return filename

        else:
            # Otherwise, treat lora_url as a Hugging Face repo ID
            # (e.g. "histin116/Hunyuan-Social-Fashion-Lora")
            repo_id = lora_url.strip()
            if "/" not in repo_id:
                raise ValueError(
                    f"Invalid Hugging Face repo ID '{repo_id}', format should be 'user/repo'."
                )

            api = HfApi()
            try:
                files = api.list_repo_files(repo_id)
            except Exception as e:
                raise ValueError(
                    f"Failed to access Hugging Face repo '{repo_id}': {e}"
                ) from e

            # Find the first available .safetensors file
            safetensors_files = [f for f in files if f.endswith(".safetensors")]
            if not safetensors_files:
                raise ValueError(
                    f"No .safetensors files found in Hugging Face repo: {repo_id}"
                )

            # Take the first .safetensors file
            hf_filename = safetensors_files[0]
            # Build the direct download URL
            hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{hf_filename}"

            # Use the same logic as above to download
            filename = os.path.basename(hf_filename)
            dst_path = os.path.join(lora_dir, filename)

            resp = requests.get(hf_url)
            resp.raise_for_status()
            with open(dst_path, "wb") as f:
                f.write(resp.content)

            return filename

    def handle_replicate_weights(self, replicate_weights: Path) -> str:
        """
        Extract ONLY lora_comfyui.safetensors from the user-provided tar file
        and move it to ComfyUI/models/loras/.
        Return the final filename ("lora_comfyui.safetensors").
        """
        lora_dir = os.path.join("ComfyUI", "models", "loras")
        os.makedirs(lora_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(str(replicate_weights), "r:*") as tar:
                tar.extractall(path=temp_dir)

            # We specifically want the ComfyUI version
            comfy_lora_path = os.path.join(temp_dir, "lora_comfyui.safetensors")
            if not os.path.exists(comfy_lora_path):
                raise FileNotFoundError(
                    "No 'lora_comfyui.safetensors' found in the provided tar."
                )

            filename = "lora_comfyui.safetensors"
            dst_path = os.path.join(lora_dir, filename)
            shutil.copy2(comfy_lora_path, dst_path)

        return filename

    def make_multi_lora_node(self, workflow, lora_names, lora_strengths):
        """
        Create a HunyuanVideoLoraLoader node that handles multiple LoRAs.
        This uses the ComfyUI-HunyuanVideoMultiLora node.

        Returns the node ID.
        """
        # Find a free node ID
        node_id = 1
        while str(node_id) in workflow:
            node_id += 1

        node_id = str(node_id)

        # Create the node
        workflow[node_id] = {
            "inputs": {
                "model": ["1", 0],  # Connect to the model loader
                "lora_name": lora_names[0],
                "strength": lora_strengths[0],
                "blocks_type": "double_blocks"
            },
            "class_type": "HunyuanVideoLoraLoader",
            "_meta": {
                "title": "Hunyuan Multi LoRA Loader"
            }
        }

        # If there's a second LoRA, add it to the node parameters
        if len(lora_names) > 1 and len(lora_strengths) > 1 and lora_names[1]:
            # The second LoRA is applied using the same node but with separate parameters
            workflow[node_id]["inputs"]["lora_name_2"] = lora_names[1]
            workflow[node_id]["inputs"]["strength_2"] = lora_strengths[1]

        return node_id

    def update_workflow(
        self,
        workflow: dict[str, Any],
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        flow_shift: int,
        seed: int,
        force_offload: bool,
        denoise_strength: float,
        num_frames: int,
        lora_names: List[str],
        lora_strengths: List[float],
        frame_rate: int = 16,
        crf: int = 19,
        enhance_weight: float = 0.3,
        enhance_single: bool = True,
        enhance_double: bool = True,
        enhance_start: float = 0.0,
        enhance_end: float = 1.0,
        scheduler: str = "DPMSolverMultistepScheduler",
    ):
        """
        Update the t2v-lora-updated.json workflow with user-selected parameters.
        Uses ComfyUI-HunyuanVideoMultiLora plugin for multiple LoRAs support.
        """
        # Create a new multi-lora node if we have valid LoRAs
        if lora_names and lora_names[0]:
            multi_lora_node_id = self.make_multi_lora_node(workflow, lora_names, lora_strengths)
            # Update the model loader to use the multi-lora node
            workflow["1"]["inputs"]["lora"] = [multi_lora_node_id, 0]

            # Remove the original lora node if it exists
            if "41" in workflow:
                del workflow["41"]

        # Node 3: HyVideoSampler
        workflow["3"]["inputs"]["width"] = width
        workflow["3"]["inputs"]["height"] = height
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["embedded_guidance_scale"] = guidance_scale
        workflow["3"]["inputs"]["flow_shift"] = flow_shift
        workflow["3"]["inputs"]["seed"] = seed
        workflow["3"]["inputs"]["force_offload"] = 1 if force_offload else 0
        workflow["3"]["inputs"]["denoise_strength"] = denoise_strength
        workflow["3"]["inputs"]["num_frames"] = num_frames
        workflow["3"]["inputs"]["scheduler"] = scheduler

        # Node 30: HyVideoTextEncode
        workflow["30"]["inputs"]["prompt"] = prompt
        workflow["30"]["inputs"]["force_offload"] = (
            "bad quality video" if force_offload else " "
        )

        # Node 34: VHS_VideoCombine
        workflow["34"]["inputs"]["frame_rate"] = frame_rate
        workflow["34"]["inputs"]["crf"] = crf
        workflow["34"]["inputs"]["save_output"] = True

        # Node 42: HyVideoEnhanceAVideo
        workflow["42"]["inputs"]["weight"] = enhance_weight
        workflow["42"]["inputs"]["single_blocks"] = enhance_single
        workflow["42"]["inputs"]["double_blocks"] = enhance_double
        workflow["42"]["inputs"]["start_percent"] = enhance_start
        workflow["42"]["inputs"]["end_percent"] = enhance_end

    def validate_dimension(self, n: int, multiple: int = 16) -> int:
        """
        Adjusts dimension to nearest multiple of 16 (required by model).
        """
        return ((n + multiple - 1) // multiple) * multiple

    def validate_enhance_params(self, start: float, end: float) -> tuple[float, float]:
        """
        Validates and adjusts enhancement start/end percentages.
        - Must be between 0.0 and 1.0
        - Start must be less than end
        """
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))

        if start >= end:
            print(
                f"⚠️  Adjusted enhance_end from {end} to {start + 0.1} to ensure it's greater than enhance_start"
            )
            end = min(1.0, start + 0.1)

        return start, end

    def validate_frame_count(self, n: int) -> int:
        """
        Adjusts frame count to nearest valid value where (n-1) is divisible by 4.
        Valid values follow pattern: 4n + 1 where n ≥ 0 (e.g., 1, 5, 9, 13, 17, 21, 25, 29, 33, ...)
        """
        if n <= 1:
            return 1
        # Subtract 1, round to nearest multiple of 4, add 1 back
        return (((n - 1) + 2) // 4 * 4) + 1

    def predict(
        self,
        # -------------------------------------------
        # 1. PROMPT & LoRAs
        # -------------------------------------------
        prompt: str = Input(
            default="",
            description="The text prompt describing your video scene.",
        ),
        lora_url: str = Input(
            default="",
            description="A URL pointing to your primary LoRA .safetensors file or a Hugging Face repo (e.g. 'user/repo' - uses the first .safetensors file).",
        ),
        lora_strength: float = Input(
            default=1.0,
            ge=-10.0,
            le=10.0,
            description="Scale/strength for your primary LoRA.",
        ),
        lora_url2: str = Input(
            default="",
            description="A URL pointing to your secondary LoRA .safetensors file or a Hugging Face repo. Leave empty to use only one LoRA.",
        ),
        lora_strength2: float = Input(
            default=0.5,
            ge=-10.0,
            le=10.0,
            description="Scale/strength for your secondary LoRA.",
        ),
        # -------------------------------------------
        # 2. SAMPLING CONTROLS
        # -------------------------------------------
        scheduler: str = Input(
            default="DPMSolverMultistepScheduler",
            choices=[
                "FlowMatchDiscreteScheduler",
                "SDE-DPMSolverMultistepScheduler",
                "DPMSolverMultistepScheduler",
                "SASolverScheduler",
                "UniPCMultistepScheduler",
            ],
            description="Algorithm used to generate the video frames.",
        ),
        steps: int = Input(
            default=50,
            ge=1,
            le=150,
            description="Number of diffusion steps.",
        ),
        guidance_scale: float = Input(
            default=6.0,
            ge=0.0,
            le=30.0,
            description="Overall influence of text vs. model.",
        ),
        flow_shift: int = Input(
            default=9,
            ge=0,
            le=20,
            description="Video continuity factor (flow).",
        ),
        # -------------------------------------------
        # 3. VIDEO DIMENSIONS & FRAMES
        # -------------------------------------------
        num_frames: int = Input(
            default=33,
            ge=1,
            le=1440,
            description="How many frames (duration) in the resulting video.",
        ),
        width: int = Input(
            default=640,
            ge=64,
            le=1536,
            description="Width for the generated video.",
        ),
        height: int = Input(
            default=360,
            ge=64,
            le=1024,
            description="Height for the generated video.",
        ),
        # -------------------------------------------
        # 4. ADVANCED CONTROLS
        # -------------------------------------------
        denoise_strength: float = Input(
            default=1.0,
            ge=0.0,
            le=2.0,
            description="Controls how strongly noise is applied each step.",
        ),
        force_offload: bool = Input(
            default=True,
            description="Whether to force model layers offloaded to CPU.",
        ),
        # -------------------------------------------
        # 5. OUTPUT ENCODING
        # -------------------------------------------
        frame_rate: int = Input(
            default=16,
            ge=1,
            le=60,
            description="Video frame rate.",
        ),
        crf: int = Input(
            default=19,
            ge=0,
            le=51,
            description="CRF (quality) for H264 encoding. Lower values = higher quality.",
        ),
        # -------------------------------------------
        # 6. POST-PROCESS ENHANCING
        # -------------------------------------------
        enhance_weight: float = Input(
            default=0.3,
            ge=0.0,
            le=2.0,
            description="Strength of the video enhancement effect.",
        ),
        enhance_single: bool = Input(
            default=True,
            description="Apply enhancement to individual frames.",
        ),
        enhance_double: bool = Input(
            default=True,
            description="Apply enhancement across frame pairs.",
        ),
        enhance_start: float = Input(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="When to start enhancement in the video. Must be less than enhance_end.",
        ),
        enhance_end: float = Input(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="When to end enhancement in the video. Must be greater than enhance_start.",
        ),
        # -------------------------------------------
        # 7. REPRODUCIBILITY: SEED
        # -------------------------------------------
        seed: int = seed_helper.predict_seed(),
        replicate_weights: Path = Input(
            default=None,
            description="A .tar file containing LoRA weights from replicate.",
        ),
    ) -> Path:
        """
        Create a video using HunyuanVideo with either:
          • replicate_weights tar (preferred if provided)
          • direct lora_url(s) (HTTP link(s) or Hugging Face repo ID(s))

        Uses the ComfyUI-HunyuanVideoMultiLora extension to support multiple LoRAs.
        """
        # Convert user seed to a valid integer
        seed = seed_helper.generate(seed)

        # Validate and adjust dimensions
        original_width, original_height = width, height
        width = self.validate_dimension(width)
        height = self.validate_dimension(height)
        if width != original_width or height != original_height:
            print(
                f"⚠️  Adjusted dimensions from {original_width}x{original_height} to {width}x{height} to satisfy model requirements"
            )

        # Validate and adjust frame count
        original_frames = num_frames
        num_frames = self.validate_frame_count(num_frames)
        if num_frames != original_frames:
            print(
                f"⚠️  Adjusted frame count from {original_frames} to {num_frames} to satisfy model requirements"
            )

        # Validate enhance parameters
        enhance_start, enhance_end = self.validate_enhance_params(
            enhance_start, enhance_end
        )

        # Double check that our frame count is valid
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"Internal error: frame count {num_frames} is invalid after adjustment. "
                f"Please report this bug."
            )

        # 1. Clean up previous runs
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # 2. Set up LoRAs array
        lora_names = []
        lora_strengths = []

        # 2a. Handle primary LoRA (either from replicate_weights or lora_url)
        if replicate_weights is not None:
            # Use replicate tar (prefer the comfyui version)
            lora_names.append(self.handle_replicate_weights(replicate_weights))
            lora_strengths.append(lora_strength)
        elif lora_url:
            # Use the remote url or huggingface repo
            lora_names.append(self.copy_lora_file(lora_url))
            lora_strengths.append(lora_strength)
        else:
            # No primary LoRA
            lora_names.append("")
            lora_strengths.append(0.0)

        # 2b. Handle secondary LoRA if provided
        if lora_url2:
            lora_names.append(self.copy_lora_file(lora_url2))
            lora_strengths.append(lora_strength2)
            print(f"Loaded second LoRA: {lora_names[-1]}")
        else:
            # No secondary LoRA
            lora_names.append("")
            lora_strengths.append(0.0)

        # 3. Load the updated workflow JSON
        with open(api_json_file, "r") as f:
            workflow = json.loads(f.read())

        # 4. Fill in user parameters with empty LoRA settings for now
        self.update_workflow(
            workflow=workflow,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            seed=seed,
            force_offload=force_offload,
            denoise_strength=denoise_strength,
            num_frames=num_frames,
            lora_names=["", ""],  # Use empty for now
            lora_strengths=[0.0, 0.0],
            frame_rate=frame_rate,
            crf=crf,
            enhance_weight=enhance_weight,
            enhance_single=enhance_single,
            enhance_double=enhance_double,
            enhance_start=enhance_start,
            enhance_end=enhance_end,
            scheduler=scheduler,
        )

        # 5. Load the workflow - weight downloader won't see the real LoRAs yet
        wf = self.comfyUI.load_workflow(workflow)

        # 6. Now update the workflow with the real LoRA settings
        self.update_workflow(
            workflow=wf,
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            seed=seed,
            force_offload=force_offload,
            denoise_strength=denoise_strength,
            num_frames=num_frames,
            lora_names=lora_names,
            lora_strengths=lora_strengths,
            frame_rate=frame_rate,
            crf=crf,
            enhance_weight=enhance_weight,
            enhance_single=enhance_single,
            enhance_double=enhance_double,
            enhance_start=enhance_start,
            enhance_end=enhance_end,
            scheduler=scheduler,
        )

        # 7. Run the workflow
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # 8. Retrieve final output
        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        if not output_files:
            raise RuntimeError("No output video was generated.")
        return output_files[0]