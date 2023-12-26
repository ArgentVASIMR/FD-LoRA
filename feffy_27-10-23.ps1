# https://github.com/kohya-ss/sd-scripts

# Based on the powershell script in Raven's LoRA Training Rentry: https://rentry.org/59xed3
# Last edited 2023-10-08 (feffy)
# - Change default optimizer to prodigy: run `pip install prodigyopt` in your venv first
# - Reduced network_dim to 32 because 128 is overkill
# - Add `--scale_weight_norms=1.0` to default flags
# - enables v_parameterization and zero_terminal_snr
# - (fix) Round bucket resolutions to multiples of 64, using ceiling on max to avoid biasing resolutions downward

# Don't be out of date! Ensure that you are using newer versions when possible.
# Ask me for updated versions: ArgentFrequencies#9944
# Alternatively, go to Furry Diffusion, look for the LoRA training thread: https://discord.gg/furrydiffusion

# Directories Config
    $image_dir = ".\.image-dir\"; # Training images folder
    $output = ".\.output\"; # Output folder for your baked LORAs.
    $reg_dir = ".\.reg-dir\"; # Regularisation images folder
    $model_dir = "C:\stable-diffusion-webui\models\Stable-diffusion"; # Path to your models folder

    $model = "MODEL.safetensors"; # Filename of the base model you wish to train on.
    $clip_skip = 1 # Set this to the clip skip that the base model was trained on. 1 for FluffyRock/Base SD, 2 for Fluffusion/NAI.

    $prompts = ""; # Direct to a text file containing your prompts. If you don't want preview images, leave this blank.

# Training Config
        $lora_name = "MyLora" # Name of LoRA
        $version = "v1.0" # Version number (Completely optional, but recommended)

    # Basic Settings:
        $real_steps    = 2000 # Total number of images processed. Actual step count will be lower when effective batch size > 1
        $save_amount   = 10 # How many LoRA checkpoints to save (e.g., 2000 steps / 10 saves == 1 save every 200 steps, 10 saves in total)
        $base_res      = 640 # The "base resolution" to train at.
        $max_aspect    = 1.5 # Determines the most extreme allowed aspect ratio for bucketing.
        $batch_size    = 1 # Amount of images to process per step. Speeds things up, but demands VRAM.

    # Advanced Settings:
        $unet_lr       = 1.0 # Unet learning rate.
        $text_enc_lr   = 1.0 # Text encoder learning rate.
        $grad_acc_step = 1 # Accumulates gradient over multiple steps to simulate higher batch size.
        $net_dim       = 32 # Network dimensions.
        $net_alpha     = 1 # Network alpha. Leave at 1 when using Prodigy
        $optimizer     = "Prodigy" # Valid values: "AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SDGNesterov8bit", "DAdaptation", "AdaFactor", "Prodigy"
        $scheduler     = "cosine_with_restarts" # Valid values: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"
        $noise_offset  = 0.0 # Ugly hack to increases dynamic range of outputs. Every 0.1 dampens learning quite a bit, do more steps or higher training rates to compensate. Prefer `--zero_terminal_snr` instead
        $keep_tags     = 1 # Keeps <n> tags at the front without shuffling them. 0 if no regularization, 1 with regularization, multi concepts may need > 1

# ==================================================================================
# BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING!
# ==================================================================================
$eff_batch_size = $batch_size * $grad_acc_step

# Scale learn rates by batch size if not using Prodigy
if ($optimizer -ine "prodigy") {
    $unet_lr = $unet_lr * $eff_batch_size # Learning rates multiplied by batch size
    $text_enc_lr = $text_enc_lr * $eff_batch_size
}

$real_steps = [int]($real_steps / $eff_batch_size) # Step count is divided by batch size. 
$save_nth_step = [int]($real_steps / $save_amount) # Save count is divided by current real steps.

# Additional lines to automatically generated named folders:
$full_name =  $lora_name + "_" + $version
$unique_output = $output + $full_name

# Bucketing res calculation [1]
if ($max_aspect -lt 1) {
    $max_aspect = 1/$max_aspect # Flip aspect ratio if it's less than 1
}
$max_bucket_res = [int]([Math]::Ceiling([Math]::Sqrt(($base_res * $base_res * $max_aspect)) / 64) * 64)
$min_bucket_res = [int]([Math]::Floor([Math]::Sqrt(($base_res * $base_res / $max_aspect)) / 64) * 64)

.\venv\scripts\activate

accelerate launch --num_cpu_threads_per_process 8 train_network.py `
    --logging_dir="logs" --log_prefix="$lora_name" `
    --network_module="networks.lora" `
    --max_data_loader_n_workers=1 --persistent_data_loader_workers `
    --caption_extension=".txt" --shuffle_caption --keep_tokens="$keep_tags" --max_token_length=225 `
    --prior_loss_weight=1 `
    --mixed_precision="fp16" --save_precision="fp16" `
    --xformers --cache_latents `
    --save_model_as=safetensors `
    --train_data_dir="$image_dir" --output_dir="$unique_output" --reg_data_dir="$reg_dir" --pretrained_model_name_or_path="$model_dir\$model" `
    --output_name="$full_name"_ `
    --learning_rate="$unet_lr" --text_encoder_lr="$text_enc_lr" `
    --max_train_steps="$real_steps" --save_every_n_steps="$save_nth_step" `
    --resolution="$base_res" `
    --enable_bucket --min_bucket_reso="$min_bucket_res" --max_bucket_reso="$max_bucket_res" `
    --train_batch_size="$batch_size" `
    --network_dim="$net_dim" --network_alpha="$net_alpha" `
    --optimizer_type="$optimizer" `
    --lr_scheduler="$scheduler" `
    --noise_offset="$noise_offset" `
    --seed=0 `
    --clip_skip="$clip_skip" `
    --sample_every_n_steps="$save_nth_step" `
    --sample_prompts="$prompts" `
    --sample_sampler="k_euler_a" `
    --gradient_accumulation_steps="$grad_acc_step" `
    --min_snr_gamma=5 `
    --scale_weight_norms=1.0 `
    --v_parameterization --zero_terminal_snr `

pause

# If you are using outdated torch, run this in a fresh powershell window (do not copy <##>):

<#
cd sd-scripts
.\venv\Scripts\activate
pip install torch==2.0.1+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118 xformers==0.0.20


#>

# If you are unsure of your torch version, run this in a fresh powershell window:

<#
cd sd-scripts
pip show torch
#>

# Sources:
# [1] == https://math.stackexchange.com/questions/2133509/how-do-i-calculate-the-length-and-width-for-a-known-area-and-ratio
