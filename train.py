import os
import shutil
import subprocess
import sys
from zipfile import ZipFile, is_zipfile
from huggingface_hub import HfApi
from cog import BaseModel, Input, Path, Secret


# We return a path to our tarred LoRA weights at the end
class TrainingOutput(BaseModel):
    weights: Path


INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_CACHE = "ckpts"
HF_UPLOAD_DIR = "hunyuan-lora-for-hf"

MODEL_FILES = ["hunyuan-video-t2v-720p.tar", "text_encoder.tar", "text_encoder_2.tar"]
BASE_URL = "https://weights.replicate.delivery/default/hunyuan-video/ckpts/"

sys.path.append("musubi-tuner")


def train(
    input_videos: Path = Input(
        description="A zip file containing videos and matching .txt captions.",
        default=None,
    ),
    epochs: int = Input(
        description="Number of training epochs (each approximately one pass over dataset).",
        default=16,
        ge=1,
        le=2000,
    ),
    rank: int = Input(
        description="LoRA rank for Hunyuan training.",
        default=32,
        ge=1,
        le=128,
    ),
    batch_size: int = Input(
        description="Batch size for training",
        default=4,
        ge=1,
        le=8,
    ),
    learning_rate: float = Input(
        description="Learning rate for training.",
        default=1e-3,
        ge=1e-5,
        le=1,
    ),
    optimizer: str = Input(
        description="Optimizer type",
        default="adamw8bit",
        choices=["adamw", "adamw8bit", "AdaFactor", "adamw16bit"],
    ),
    gradient_checkpointing: bool = Input(
        description="Enable gradient checkpointing to reduce memory usage.",
        default=True,
    ),
    timestep_sampling: str = Input(
        description="Method to sample timesteps for training.",
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift"],
    ),
    consecutive_target_frames: str = Input(
        description="The lengths of consecutive frames to extract. Each integer represents how many consecutive frames to extract using the frame extraction method. For example, '1, 13, 25' will create three separate extractions of 1 frame, 13 frames, and 25 consecutive frames.",
        default="[1, 25, 45]",
        choices=[
            "[1, 13, 25]",
            "[1, 25, 45]",
            "[1, 45, 89]",
            "[1, 13, 25, 45]",
        ],
    ),
    frame_extraction_method: str = Input(
        description="Method to extract frames from videos during training. 'head': takes first N frames, 'chunk': splits video into N-frame chunks, 'slide': extracts frames with fixed stride, 'uniform': samples N evenly-spaced frames.",
        default="head",
        choices=["head", "chunk", "slide", "uniform"],
    ),
    frame_stride: int = Input(
        description="Frame stride for 'slide' extraction method. This represents the number of frames to advance when extracting each sequence of frames, where a sequence is typically longer than the stride value (e.g., extracting 13 frames with stride 10), resulting in overlapping segments of frames. For example, with a 13-frame sequence and stride of 10, each new sequence would share 3 frames with the previous sequence.",
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
        description="Random seed (use <=0 for a random pick).",
        default=0,
    ),
    hf_repo_id: str = Input(
        description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, lucataco/flux-dev-lora. If the given repo does not exist, a new public repo will be created.",
        default=None,
    ),
    hf_token: Secret = Input(
        description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
        default=None,
    ),
) -> TrainingOutput:
    """
    Minimal Hunyuan LoRA training script using musubi-tuner.
    """

    if not input_videos:
        raise ValueError("You must provide a zip with videos & .txt captions.")

    clean_up()
    download_weights()
    seed = handle_seed(seed)
    create_train_toml(
        consecutive_target_frames, frame_extraction_method, frame_stride, frame_sample
    )
    extract_zip(input_videos, INPUT_DIR)
    cache_latents()
    cache_text_encoder_outputs(batch_size)
    run_lora_training(
        epochs,
        rank,
        optimizer,
        learning_rate,
        timestep_sampling,
        seed,
        gradient_checkpointing,
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
    print("Creating train.toml...")
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


def cache_latents():
    """Cache latents using musubi-tuner"""
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


def cache_text_encoder_outputs(batch_size: int):
    """Cache text encoder outputs"""
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


def run_lora_training(
    epochs: int,
    rank: int,
    optimizer: str,
    learning_rate: float,
    timestep_sampling: str,
    seed: int,
    gradient_checkpointing: bool,
):
    """Run LoRA training"""
    print("Running LoRA training...")
    training_args = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
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
        "--sdpa",
        "--mixed_precision",
        "bf16",
        "--fp8_base",
        "--optimizer_type",
        optimizer,
        "--learning_rate",
        str(learning_rate),
        "--max_data_loader_n_workers",
        "2",
        "--persistent_data_loader_workers",
        "--network_module",
        "networks.lora",
        "--network_dim",
        str(rank),
        "--timestep_sampling",
        timestep_sampling,
        "--discrete_flow_shift",
        "1.0",
        "--max_train_epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--output_dir",
        OUTPUT_DIR,
        "--output_name",
        "lora",
    ]
    if gradient_checkpointing:
        training_args.append("--gradient_checkpointing")

    subprocess.run(training_args, check=True)


def convert_lora_to_comfyui_format():
    """Convert LoRA to ComfyUI-compatible format"""
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
        print("Warning: lora.safetensors not found, skipping conversion.")


def archive_results() -> str:
    """Archive final results and return output path"""
    output_path = "/tmp/trained_model.tar"
    print(f"Archiving LoRA outputs to {output_path}")
    os.system(f"tar -cvf {output_path} -C {OUTPUT_DIR} .")
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
            subprocess.check_call(["pget", "-xvf", url, MODEL_CACHE])


def extract_zip(zip_path: Path, extraction_dir: str):
    """Extract videos & .txt captions from the provided zip."""
    if not is_zipfile(zip_path):
        raise ValueError("The provided input_videos must be a zip file.")

    os.makedirs(extraction_dir, exist_ok=True)
    final_videos_path = os.path.join(extraction_dir, "videos")
    os.makedirs(final_videos_path, exist_ok=True)

    file_count = 0
    with ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            # skip Mac hidden system files
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, final_videos_path)
                file_count += 1

    print(f"Extracted {file_count} total files to: {final_videos_path}")


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
