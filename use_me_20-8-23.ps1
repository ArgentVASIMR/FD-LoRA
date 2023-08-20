# https://github.com/kohya-ss/sd-scripts

# Based on the powershell script in Raven's LoRA Training Rentry: https://rentry.org/59xed3
# Last edited 20/8/23 (D/M/Y)

# Don't be out of date! Ensure that you are using newer versions when possible.
# Ask me for updated versions via Discord: argentvasimr
# Alternatively, go to Furry Diffusion, look for the LoRA training thread: https://discord.gg/furrydiffusion

# Directories Config
    $image_dir = ".\.image-dir\"; # Training images folder
    $output = ".\.output\"; # Output folder for your baked LORAs.
    $reg_dir = ".\.reg-dir\"; # Regularisation images folder
    $model_dir = "C:\stable-diffusion-webui\models\Stable-diffusion"; # Path to your models folder. Must start from C:\

    $model = "v1-5-pruned-emaonly.safetensors"; # Filename of the base model you wish to train on.
    $clip_skip = 1 # Set this to the clip skip that the base model was trained on. 1 for FluffyRock/Base SD, 2 for Fluffusion/NAI.

    $prompts = ""; # Direct to a text file containing your prompts. If you don't want preview images, leave this blank.

# Training Config
        $lora_name = "MyLora" # Name of LoRA
        $version = "v1.0" # Version number (Completely optional, but recommended)

    # Basic Settings:
        $real_steps    = 2000 # Total number of steps.
        $save_amount   = 10 # How many LoRA checkpoints to save (e.g., 2000 steps / 10 saves == 1 save every 200 steps, 10 saves in total)
        $base_res      = 640 # The "base resolution" to train at.
        $max_aspect    = 1.5 # Determines the most extreme allowed aspect ratio for bucketing.
        $batch_size    = 1 # Amount of images to process per step. Speeds things up, but demands VRAM. Also averages images into gradients

    # Learning Rate Settings:
        $unet_lr       = 1e-4 # Unet learning rate.
        $text_enc_lr   = 5e-5 # Text encoder learning rate.
        $lr_multiplier = 1 # Can be used to change both LRs by roughly the same relative amount. 1 == 100% of normal.

    # Other Settings:
        $grad_acc_step = 1 # Accumulates <N> images into each gradient update. Can make training more reliable and successful, if used correctly.
        $net_dim       = 32 # Network dimensions.
        $net_alpha     = 0 # Network alpha.
        $optimizer     = "AdamW8bit" # Valid values: "AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SDGNesterov8bit", "DAdaptation", "AdaFactor"
        $scheduler     = "cosine" # Valid values: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"
        $noise_offset  = 0.0 # Increases dynamic range of outputs. Every 0.1 dampens learning quite a bit, do more steps or higher training rates to compensate.
        $keep_tags     = 0 # Keeps <N> tags at the front without shuffling them. 0 if no regularization, 1 with regularization, multi concepts may need > 1. Kohya's official name is "keep tokens".

# ========================================================================================
#    BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING!
# ========================================================================================
$eff_batch_size = $batch_size * $grad_acc_step

# Batch size must be accounted for properly:
$unet_lr = $unet_lr * $eff_batch_size # Learning rates multiplied by batch size
$text_enc_lr = $text_enc_lr * $eff_batch_size

$real_steps = [int]($real_steps / $eff_batch_size) # Step count is divided by batch size. 
$save_nth_step = [int]($real_steps / $save_amount) # Save count is divided by current real steps.

# Additional lines to automatically generated named folders:
$full_name =  $lora_name + "_" + $version
$unique_output = $output + $full_name

# Bucketing res calculation [1]
if ($max_aspect -lt 1) {
    $max_aspect = 1/$max_aspect # Flip aspect ratio if it's less than 1
}
$max_bucket_res = [int]([Math]::Sqrt(([Math]::Pow($base_res,2) * $max_aspect)))
$min_bucket_res = [int]([Math]::Sqrt(([Math]::Pow($base_res,2) / $max_aspect)))

# Apply LR multiplier
$unet_lr = $unet_lr*$lr_multiplier
$text_enc_lr = $text_enc_lr*$lr_multiplier

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
    --learning_rate="$unet_lr" --unet_lr="$unet_lr" --text_encoder_lr="$text_enc_lr" `
    --max_train_steps="$real_steps" --save_every_n_steps="$save_nth_step" `
    --resolution="$base_res" `
    --enable_bucket --min_bucket_reso="$min_bucket_res" --max_bucket_reso="$max_bucket_res" `
    --train_batch_size="$batch_size" `
    --network_dim="$net_dim" --network_alpha="$net_alpha" `
# For "Prodigy" optimiser:
#    --optimizer_args "safeguard_warmup=True" "use_bias_correction=True" "weight_decay=0.01" `
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
#    --v_parameterisation `
pause

# If you are using outdated torch, run this in a fresh powershell window (do not copy <##>):

<#
cd sd-scripts
.\venv\Scripts\activate
pip install torch==2.0.0+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118  xformers==0.0.19


#>

# If you are unsure of your torch version, run this in a fresh powershell window:

<#
cd sd-scripts
pip show torch
#>

# Sources:
# [1] == https://math.stackexchange.com/questions/2133509/how-do-i-calculate-the-length-and-width-for-a-known-area-and-ratio
