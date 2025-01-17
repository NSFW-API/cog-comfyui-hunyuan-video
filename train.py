import os
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile, is_zipfile
from huggingface_hub import HfApi
from cog import BaseModel, Input, Path, Secret
from typing import Optional
import logging


# Configure logging to suppress INFO messages
logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Suppress all common loggers
loggers_to_quiet = [
    "torch",
    "accelerate",
    "transformers",
    "__main__",
    "dataset",
    "networks",
    "hunyuan_model",
    "PIL",
    "qwen_vl_utils",
    "huggingface_hub",
    "diffusers",
    "filelock",
    "safetensors",
    "xformers",
    "datasets",
    "tokenizers",
    "sageattention",
]

for logger_name in loggers_to_quiet:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Suppress third-party warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# We return a path to our tarred LoRA weights at the end
class TrainingOutput(BaseModel):
    weights: Path


# [ADDED] Constants for Qwen2-VL model
QWEN_MODEL_CACHE = "qwen_checkpoints"
QWEN_MODEL_URL = (
    "https://weights.replicate.delivery/default/qwen/Qwen2-VL-7B-Instruct/model.tar"
)

INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_CACHE = "ckpts"
HF_UPLOAD_DIR = "hunyuan-lora-for-hf"

MODEL_FILES = ["hunyuan-video-t2v-720p.tar", "text_encoder.tar", "text_encoder_2.tar"]
BASE_URL = "https://weights.replicate.delivery/default/hunyuan-video/ckpts/"

sys.path.append("musubi-tuner")


def train(
    input_videos: Path = Input(
        description="A zip file containing videos that will be used for training. If you include captions, include them as one .txt file per video, e.g. my-video.mp4 should have a caption file named my-video.txt. If you don't include captions, you can use autocaptioning (enabled by default).",
        default=None,
    ),
    trigger_word: str = Input(
        description="The trigger word refers to the object, style or concept you are training on. Pick a string that isn't a real word, like TOK or something related to what's being trained, like STYLE3D. The trigger word you specify here will be associated with all videos during training. Then when you use your LoRA, you can include the trigger word in prompts to help activate the LoRA.",
        default="TOK",
    ),
    autocaption: bool = Input(
        description="Automatically caption videos using QWEN-VL", default=True
    ),
    autocaption_prefix: str = Input(
        description="Optional: Text you want to appear at the beginning of all your generated captions; for example, 'a video of TOK, '. You can include your trigger word in the prefix. Prefixes help set the right context for your captions.",
        default=None,
    ),
    autocaption_suffix: str = Input(
        description="Optional: Text you want to appear at the end of all your generated captions; for example, ' in the style of TOK'. You can include your trigger word in suffixes. Suffixes help set the right concept for your captions.",
        default=None,
    ),
    epochs: int = Input(
        description="Number of training epochs. Each epoch processes all your videos once. Note: If max_train_steps is set, training may end before completing all epochs.",
        default=16,
        ge=1,
        le=2000,
    ),
    max_train_steps: int = Input(
        description="Maximum number of training steps to perform. Each step processes one batch of frames. Set to -1 to train for the full number of epochs. If positive, training will stop after this many steps even if all epochs aren't complete.",
        default=-1,
        ge=-1,
        le=1_000_000,
    ),
    rank: int = Input(
        description="LoRA rank for training. Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.",
        default=32,
        ge=1,
        le=128,
    ),
    batch_size: int = Input(
        description="Batch size for training. Lower values use less memory but train slower.",
        default=4,
        ge=1,
        le=8,
    ),
    learning_rate: float = Input(
        description="Learning rate for training. If you're new to training you probably don't need to change this.",
        default=1e-3,
        ge=1e-5,
        le=1,
    ),
    optimizer: str = Input(
        description="Optimizer type for training. If you're unsure, leave as default.",
        default="adamw8bit",
        choices=["adamw", "adamw8bit", "AdaFactor", "adamw16bit"],
    ),
    timestep_sampling: str = Input(
        description="Controls how timesteps are sampled during training. 'sigmoid' (default) concentrates samples in the middle of the diffusion process. 'uniform' samples evenly across all timesteps. 'sigma' samples based on the noise schedule. 'shift' uses shifted sampling with discrete flow shift. If unsure, use 'sigmoid'.",
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift"],
    ),
    consecutive_target_frames: str = Input(
        description="The lengths of consecutive frames to extract from each video.",
        default="[1, 25, 45]",
        choices=[
            "[1, 13, 25]",
            "[1, 25, 45]",
            "[1, 45, 89]",
            "[1, 13, 25, 45]",
        ],
    ),
    frame_extraction_method: str = Input(
        description="Method to extract frames from videos during training.",
        default="head",
        choices=["head", "chunk", "slide", "uniform"],
    ),
    frame_stride: int = Input(
        description="Frame stride for 'slide' extraction method.",
        default=10,
        ge=1,
        le=100,
    ),
    frame_sample: int = Input(
        description="Number of samples for 'uniform' extraction method.",
        default=4,
        ge=1,
        le=20,
    ),
    seed: int = Input(
        description="Random seed for training. Use <=0 for random.",
        default=0,
    ),
    hf_repo_id: str = Input(
        description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, username/my-video-lora. If the given repo does not exist, a new public repo will be created.",
        default=None,
    ),
    hf_token: Secret = Input(
        description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
        default=None,
    ),
) -> TrainingOutput:
    """Minimal Hunyuan LoRA training script using musubi-tuner"""
    print("\n=== 🎥 Hunyuan Video LoRA Training ===")
    print(f"📊 Configuration:")
    print(f"  • Input: {input_videos}")
    print(f"  • Training:")
    print(f"    - Epochs: {epochs}")
    if max_train_steps > 0:
        print(f"    - Max Steps: {max_train_steps}")
    print(f"    - LoRA Rank: {rank}")
    print(f"    - Learning Rate: {learning_rate}")
    print(f"    - Batch Size: {batch_size}")
    if autocaption:
        print(f"  • Auto-captioning:")
        print(f"    - Enabled: {autocaption}")
        print(f"    - Trigger Word: {trigger_word}")
        if autocaption_prefix:
            print(f"    - Prefix: {autocaption_prefix}")
        if autocaption_suffix:
            print(f"    - Suffix: {autocaption_suffix}")
    print("=====================================\n")

    if not input_videos:
        raise ValueError(
            "You must provide a zip with videos & optionally .txt captions."
        )

    clean_up()
    download_weights()
    seed = handle_seed(seed)
    create_train_toml(
        consecutive_target_frames, frame_extraction_method, frame_stride, frame_sample
    )
    extract_zip(
        input_videos,
        INPUT_DIR,
        autocaption=autocaption,
        trigger_word=trigger_word,
        autocaption_prefix=autocaption_prefix,
        autocaption_suffix=autocaption_suffix,
    )
    cache_latents()
    cache_text_encoder_outputs(batch_size)
    run_lora_training(
        epochs=epochs,
        rank=rank,
        optimizer=optimizer,
        learning_rate=learning_rate,
        timestep_sampling=timestep_sampling,
        seed=seed,
        max_train_steps=max_train_steps,
    )
    convert_lora_to_comfyui_format()
    output_path = archive_results()

    if hf_token and hf_repo_id:
        os.makedirs(HF_UPLOAD_DIR, exist_ok=True)
        shutil.move(
            os.path.join(OUTPUT_DIR, "lora.safetensors"),
            HF_UPLOAD_DIR / Path("lora.safetensors"),
        )
        handle_hf_upload(hf_repo_id, hf_token)

    return TrainingOutput(weights=Path(output_path))


def handle_seed(seed: int) -> int:
    """Handle random seed logic"""
    if seed <= 0:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    return seed


def create_train_toml(
    consecutive_target_frames: str,
    frame_extraction_method: str,
    frame_stride: int,
    frame_sample: int,
):
    """Create train.toml configuration file"""
    print("\n=== 📝 Creating Training Configuration ===")
    with open("train.toml", "w") as f:
        target_frames = consecutive_target_frames
        if frame_extraction_method == "chunk":
            # Strip out 1 from target frames for chunk extraction
            target_frames = consecutive_target_frames
            if target_frames.startswith("[1, "):
                target_frames = "[" + target_frames[4:]

        config = f"""[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "./input/videos"
cache_directory = "./input/cache_directory"
target_frames = {target_frames}
frame_extraction = "{frame_extraction_method}"
"""
        if frame_extraction_method == "slide":
            config += f"frame_stride = {frame_stride}\n"
        elif frame_extraction_method == "uniform":
            config += f"frame_sample = {frame_sample}\n"
        f.write(config)
        print("Training config:\n==================\n")
        print(config)
        print("\n==================\n")
    print("✅ Configuration file created")
    print("=====================================")


def cache_latents():
    """Cache latents using musubi-tuner"""
    print("\n=== 💾 Caching Video Latents ===")
    latent_args = [
        "python",
        "musubi-tuner/cache_latents.py",
        "--dataset_config",
        "train.toml",
        "--vae",
        os.path.join(MODEL_CACHE, "hunyuan-video-t2v-720p/vae/pytorch_model.pt"),
        "--vae_chunk_size",
        "32",
        "--vae_tiling",
    ]
    subprocess.run(latent_args, check=True)
    print("✅ Latents cached successfully")
    print("=====================================")


def cache_text_encoder_outputs(batch_size: int):
    """Cache text encoder outputs"""
    print("\n=== 💭 Caching Text Encodings ===")
    text_encoder_args = [
        "python",
        "musubi-tuner/cache_text_encoder_outputs.py",
        "--dataset_config",
        "train.toml",
        "--text_encoder1",
        os.path.join(MODEL_CACHE, "text_encoder"),
        "--text_encoder2",
        os.path.join(MODEL_CACHE, "text_encoder_2"),
        "--batch_size",
        str(batch_size),
    ]
    subprocess.run(text_encoder_args, check=True)
    print("✅ Text encodings cached")
    print("=====================================")


def run_lora_training(
    epochs: int,
    rank: int,
    optimizer: str,
    learning_rate: float,
    timestep_sampling: str,
    seed: int,
    max_train_steps: int = -1,  # Keep same interface as train()
):
    """Run LoRA training with optional step limit"""
    print("\n=== 🚀 Starting LoRA Training ===")

    # Convert negative to zero for hv_train_network.py
    actual_max_steps = max(0, max_train_steps)

    if actual_max_steps > 0:
        print(f"• Max Train Steps: {actual_max_steps}")
    else:
        print(f"• Epochs: {epochs}")
    print("=====================================")

    training_args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "8",
        "--mixed_precision",
        "bf16",
        "musubi-tuner/hv_train_network.py",
        "--dit",
        os.path.join(
            MODEL_CACHE,
            "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        ),
        "--dataset_config",
        "train.toml",
        "--flash_attn",
        "--mixed_precision",
        "bf16",
        "--fp8_base",
        "--optimizer_type",
        optimizer,
        "--learning_rate",
        str(learning_rate),
        "--max_data_loader_n_workers",
        "16",
        "--persistent_data_loader_workers",
        "--network_module",
        "networks.lora",
        "--network_dim",
        str(rank),
        "--timestep_sampling",
        timestep_sampling,
        "--discrete_flow_shift",
        "1.0",
        "--seed",
        str(seed),
        "--output_dir",
        OUTPUT_DIR,
        "--output_name",
        "lora",
        "--gradient_checkpointing",
    ]

    # Only add one of max_train_epochs or max_train_steps
    if actual_max_steps > 0:
        training_args.extend(["--max_train_steps", str(actual_max_steps)])
    else:
        training_args.extend(["--max_train_epochs", str(epochs)])

    subprocess.run(training_args, check=True)
    print("\n✨ Training Complete!")
    print("=====================================")


def convert_lora_to_comfyui_format():
    """Convert LoRA to ComfyUI-compatible format"""
    print("\n=== 🔄 Converting LoRA Format ===")
    original_lora_path = os.path.join(OUTPUT_DIR, "lora.safetensors")
    if os.path.exists(original_lora_path):
        converted_lora_path = os.path.join(OUTPUT_DIR, "lora_comfyui.safetensors")
        print(
            f"Converting from {original_lora_path} -> {converted_lora_path} (ComfyUI format)"
        )
        convert_args = [
            "python",
            "musubi-tuner/convert_lora.py",
            "--input",
            original_lora_path,
            "--output",
            converted_lora_path,
            "--target",
            "other",  # "other" -> diffusers style (ComfyUI)
        ]
        subprocess.run(convert_args, check=True)
    else:
        print("⚠️  Warning: lora.safetensors not found, skipping conversion.")
    print("✅ Converted to ComfyUI format")
    print("=====================================")


def archive_results() -> str:
    """Archive final results and return output path"""
    print("\n=== 📦 Archiving Results ===")
    output_path = "/tmp/trained_model.tar"

    print(f"Archiving LoRA outputs to {output_path}")
    os.system(f"tar -cvf {output_path} -C {OUTPUT_DIR} .")
    print(f"✅ Results archived to: {output_path}")
    print("=====================================")
    return output_path


def clean_up():
    """Removes INPUT_DIR, OUTPUT_DIR, and HF_UPLOAD_DIR if they exist."""
    for dir in [INPUT_DIR, OUTPUT_DIR, HF_UPLOAD_DIR]:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def download_weights():
    """Download base Hunyuan model weights if not already cached."""
    os.makedirs(MODEL_CACHE, exist_ok=True)
    for model_file in MODEL_FILES:
        filename_no_ext = model_file.split(".")[0]
        dest_path = os.path.join(MODEL_CACHE, filename_no_ext)
        if not os.path.exists(dest_path):
            url = BASE_URL + model_file
            print(f"Downloading {url} to {MODEL_CACHE}")
            subprocess.check_call(["pget", "-xf", url, MODEL_CACHE])


def autocaption_videos(
    videos_path: str,
    video_files: set,
    caption_files: set,
    trigger_word: Optional[str] = None,
    autocaption_prefix: Optional[str] = None,
    autocaption_suffix: Optional[str] = None,
) -> set:
    """Generate captions for videos that don't have matching .txt files."""
    videos_without_captions = video_files - caption_files
    if not videos_without_captions:
        return caption_files

    print("\n=== 🤖 Auto-captioning Videos ===")
    print(f"Found {len(videos_without_captions)} videos without captions")
    model, processor = setup_qwen_model()

    new_caption_files = caption_files.copy()
    for i, vid_name in enumerate(videos_without_captions, 1):
        mp4_path = os.path.join(videos_path, vid_name + ".mp4")
        if os.path.exists(mp4_path):
            print(f"\n[{i}/{len(videos_without_captions)}] 🎥 {vid_name}.mp4")

            # Use absolute path
            abs_path = os.path.abspath(mp4_path)

            # Build caption components
            prefix = f"{autocaption_prefix.strip()} " if autocaption_prefix else ""
            suffix = f" {autocaption_suffix.strip()}" if autocaption_suffix else ""
            trigger = f"{trigger_word} " if trigger_word else ""

            # Prepare messages format with customized prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": abs_path,
                        },
                        {
                            "type": "text",
                            "text": "Describe this video clip in detail, focusing on the key visual elements, actions, and overall scene.",
                        },
                    ],
                }
            ]

            # Process inputs
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                # Import qwen utils here to avoid circular imports
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                print("\nGenerating caption...")
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                caption = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Combine prefix, trigger, caption, and suffix
                final_caption = f"{prefix}{trigger}{caption.strip()}{suffix}"
                print("\n📝 Generated Caption:")
                print("--------------------")
                print(f"{final_caption}")
                print("--------------------")

            except Exception as e:
                print(f"\n⚠️  Warning: Failed to autocaption {vid_name}.mp4")
                print(f"Error: {str(e)}")
                final_caption = (
                    f"{prefix}{trigger}A video clip named {vid_name}{suffix}"
                )
                print("\n📝 Using fallback caption:")
                print("--------------------")
                print(f"{final_caption}")
                print("--------------------")

            # Save caption
            txt_path = os.path.join(videos_path, vid_name + ".txt")
            with open(txt_path, "w") as f:
                f.write(final_caption.strip() + "\n")
            new_caption_files.add(vid_name)
            print(f"✅ Saved to: {txt_path}")

    # Clean up QWEN model
    print("\n=== 🧹 Cleaning Up ===")
    del model
    del processor
    import torch

    torch.cuda.empty_cache()

    print(f"✨ Successfully processed {len(videos_without_captions)} videos!")
    print("=====================================")

    return new_caption_files


def extract_zip(
    zip_path: Path,
    extraction_dir: str,
    autocaption: bool = True,
    trigger_word: Optional[str] = None,
    autocaption_prefix: Optional[str] = None,
    autocaption_suffix: Optional[str] = None,
):
    """
    Extract videos & .txt captions from the provided zip.
    If autocaption is True, generate captions for missing videos.
    """
    if not is_zipfile(zip_path):
        raise ValueError("The provided input_videos must be a zip file.")

    # Setup directories
    os.makedirs(extraction_dir, exist_ok=True)
    final_videos_path = os.path.join(extraction_dir, "videos")
    os.makedirs(final_videos_path, exist_ok=True)

    # Extract and track files
    video_files = set()
    caption_files = set()
    file_count = 0

    with ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                base_name = os.path.basename(file_info.filename)
                if base_name:
                    if base_name.endswith(".mp4"):
                        video_files.add(os.path.splitext(base_name)[0])
                    elif base_name.endswith(".txt"):
                        caption_files.add(os.path.splitext(base_name)[0])
                zip_ref.extract(file_info, final_videos_path)
                file_count += 1

    # Flatten directory structure
    for root, dirs, files in os.walk(final_videos_path):
        if root == final_videos_path:
            continue
        for f in files:
            old_path = os.path.join(root, f)
            new_path = os.path.join(final_videos_path, f)
            shutil.move(old_path, new_path)

    # Clean up empty directories
    for root, dirs, files in os.walk(final_videos_path, topdown=False):
        if root != final_videos_path:
            try:
                os.rmdir(root)
            except OSError:
                pass

    # Handle autocaptioning if needed
    if autocaption:
        caption_files = autocaption_videos(
            final_videos_path,
            video_files,
            caption_files,
            trigger_word,
            autocaption_prefix,
            autocaption_suffix,
        )

    # Final validation
    videos_without_captions = video_files - caption_files
    captions_without_videos = caption_files - video_files

    if not video_files:
        raise ValueError("No video files found in zip!")
    if not caption_files:
        raise ValueError(
            "No caption files found in zip (and autocaption didn't generate any)!"
        )
    if not (video_files & caption_files):
        raise ValueError(
            "No matching video-caption pairs found after checking or generating captions!"
        )

    # Report results
    print(f"✅ Extracted {file_count} files to: {final_videos_path}")
    print(f"📊 Status:")
    print(f"  • Valid Pairs: {len(video_files & caption_files)} video-caption pairs")
    if videos_without_captions:
        print(f"  • ⚠️  Missing Captions: {len(videos_without_captions)} videos")
    if captions_without_videos:
        print(f"  • ⚠️  Orphaned Captions: {len(captions_without_videos)} files")
    print("=====================================")


def handle_hf_upload(hf_repo_id: str, hf_token: Secret):
    print(f"HF Token: {hf_token}")
    print(f"HF Repo ID: {hf_repo_id}")
    if hf_token is not None and hf_repo_id is not None:
        try:
            title = handle_hf_readme(hf_repo_id)
            print(f"Uploading to Hugging Face: {hf_repo_id}")
            api = HfApi()

            repo_url = api.create_repo(
                hf_repo_id,
                private=False,
                exist_ok=True,
                token=hf_token.get_secret_value(),
            )

            print(f"HF Repo URL: {repo_url}")

            # Rename lora.safetensors to hunyuan-[title].safetensors
            old_path = HF_UPLOAD_DIR / Path("lora.safetensors")
            new_name = title.lower()
            if not new_name.startswith("hunyuan"):
                new_name = f"hunyuan-{new_name}"
            new_path = HF_UPLOAD_DIR / Path(f"{new_name}.safetensors")
            os.rename(old_path, new_path)

            api.upload_folder(
                repo_id=hf_repo_id,
                folder_path=HF_UPLOAD_DIR,
                repo_type="model",
                use_auth_token=hf_token.get_secret_value(),
            )
        except Exception as e:
            print(f"Error uploading to Hugging Face: {str(e)}")


def handle_hf_readme(hf_repo_id: str) -> str:
    readme_path = HF_UPLOAD_DIR / Path("README.md")
    license_path = Path("hf-lora-readme-template.md")
    shutil.copy(license_path, readme_path)

    content = readme_path.read_text()

    repo_parts = hf_repo_id.split("/")
    if len(repo_parts) > 1:
        title = repo_parts[1].replace("-", " ").title()
        content = content.replace("[title]", title)
    else:
        title = hf_repo_id
        content = content.replace("[title]", title)

    print("HF readme content:\n==================\n")
    print(content)
    print("\n==================\n")
    readme_path.write_text(content)
    return title.replace(" ", "-")


def setup_qwen_model():
    """Download and setup Qwen2-VL model for auto-captioning"""
    # TODO: use download_weights() instead
    if not os.path.exists(QWEN_MODEL_CACHE):
        print(f"Downloading Qwen2-VL model to {QWEN_MODEL_CACHE}")
        start = time.time()
        subprocess.check_call(["pget", "-xf", QWEN_MODEL_URL, QWEN_MODEL_CACHE])
        print(f"Download took: {time.time() - start:.2f}s")

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print("\nLoading QWEN model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_CACHE,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_CACHE)

    return model, processor
