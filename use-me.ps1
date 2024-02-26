# This script is intended for use with sd-scripts, please install that if you haven't already:
# https://github.com/kohya-ss/sd-scripts

# Please visit the Furry Diffusion LoRA repo!
# https://github.com/ArgentVASIMR/FD-lora

# Main contributors (discord): 
    # argentvasimr
    # feffy
    # mmmmmmmm
# Last edited 26-02-2024 (argentvasimr)
# To implement:
    # Space-tolerant directories
    # Fix unicode issue (if fixable)

# For assistance, please ask in the LoRA training thread on Furry Diffusion: https://discord.com/channels/1019133813105905664/1065749961083469884

# Directories:
    $dataset_dir = ".\.dataset"
    $output_dir = ".\.output"
    $reg_dir = ".\.reg"
    $checkpoint_dir = "C:\your\models\folder\here"
    $prompts = ""; # Leave blank if you don't want sample images

# Model Settings:
    $checkpoint     = "v1-5-pruned-emaonly.safetensors"
    $clip_skip      = 1
    $v_prediction   = $false
    $sdxl           = $false

# Training Config:
    # Output Settings:
        $lora_name      = "myFirstLoRA"
        $version        = "prototype"
        $comment        = "This is my first LoRA"
        $save_amount    = 10

    # Dataset Treatment:
        $base_res       = 512
        $max_aspect     = 1.5
        $bucket_step    = 8
        $flip_aug       = $true
        $keep_tags      = 1

    # Steps:
        $total_steps    = 2000
        $batch_size     = 1
        $grad_acc_step  = 1
        $warmup         = 200 # If $warmup_type is "percent", must be less than 1.0, otherwise it is step count
        $warmup_type    = "steps" # "steps", "steps_batch", "percent"

    # Learning Rate:
        $unet_lr        = 1e-4
        $text_enc_lr    = 5e-5
        $lr_scheduler   = "cosine" # Recommended options are "cosine", "linear".

    # Network:
        $net_dim        = 32
        $net_alpha      = 32
        $optimiser      = "adamw" # Recommended options are "adamw", "adamw8bit", "dadaptadam".

    # Performance:
        $grad_checkpt   = $false
        $full_fp16      = $false
        $fp8_base       = $false
        $latents_disk   = $false

# =============================================================================================
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# =============================================================================================

    # Debugging:
        $correct_alpha  = $false # Apply scaling to alpha, multiplying by sqrt($net_dim)
        $scale_lr       = $true # Scale learning rate by batch size.
        $is_lr_free     = $false # Manual toggle for LR-free optimisers, in case one isn't accounted for.

    # Advanced:
    # (MAY BE REMOVED FROM CONFIG OUTRIGHT IN FUTURE IF NO ADDITIONAL CHANGES ARE RECOMMENDED)
        $weight_decay   = 0.01
        $seed           = 0
        $cap_dropout    = 0
        $net_dropout    = 0.1
        $scale_weight   = 1

# Declaring/setting variables for later
    $run_script = "train_network.py"
    $extra = @()
    $opt_args = @()

# Directories
    if ($prompts -ne "") {
        $extra += "--sample_prompts=$prompts","--sample_sampler=euler","--sample_every_n_steps=$save_nth_step"
    }

# Model Settings
    $checkpoint_dir_full = "$checkpoint_dir\$checkpoint"
    if ($v_prediction -eq $true) {
        $extra += "--v_parameterization","--zero_terminal_snr"
        $noise_offset = 0.0
    } else {
        $noise_offset = 0.01
    }
    if ($sdxl -eq $true) {
        $run_script = "SDXL_train_network.py"
    }

# Output Settings
    if ($comment -ne ""){
        $extra += "--training_comment=$comment"
    }

# Dataset Treatment
    # Bucketing res calculations via $max_aspect [1]
    if ($max_aspect -lt 1) {$max_aspect = 1/$max_aspect} # Flip aspect ratio if it's less than 1
    $max_bucket_res = [int]([Math]::Ceiling([Math]::Sqrt(($base_res * $base_res * $max_aspect)) / 64) * 64)
    $min_bucket_res = [int]([Math]::Floor([Math]::Sqrt(($base_res * $base_res / $max_aspect)) / 64) * 64)

    if ($flip_aug -eq $true) {
        $extra += "--flip_aug"
    }

# Steps
    $total_grad_acc = $batch_size*$grad_acc_step
    $total_steps = [int]([Math]::Round($total_steps / $total_grad_acc))
    $save_nth_step = [int]([Math]::Round($total_steps / $save_amount))

    # Check if warmup is set to a valid value with the warmup type
    if (($warmup -ge 1) -and ($warmup_type -eq "percent")) {
        Write-Color "The warmup percentage is set to equal to or greater than 1.0 / 100% ($warmup)." -ForegroundColor Red
        Write-Host "Please set `$warmup to less than 1 (recommended value is 0.1 / 10%), OR set `$warmup_type to `"steps`" or `"steps_batch`"." -ForegroundColor Red
        pause
        exit
    } elseif (($warmup -lt 1) -and ($warmup_type -ne "percent")) {
        Write-Host "The warmup steps are less than 1 ($warmup)." -ForegroundColor Red
        Write-Host "Please set `$warmup to an integer greater than 1 (recommended value is 200), OR set `$warmup_type to `"percent`"." -ForegroundColor Red
        pause
        exit
    } elseif (($warmup -isnot [int]) -and ($warmup_type -ne "percent")) {
        Write-Host "The inputted warmup steps is not an integer." -ForegroundColor Red
        Write-Host "Please set an integer value for `$warmup or set `$warmup_type to `"percent`"." -ForegroundColor Red
        pause
        exit
    }

    if (($warmup_type -eq "percent") -or ($warmup_type -eq "steps_batch")){
        $warmup_steps = [int]([Math]::Round(($total_steps*$warmup)))
    } elseif ($warmup_type -eq "steps") {
        $warmup_steps = $warmup
    }
    if ($warmup -gt 0) {
        $extra += "--lr_warmup_steps=$warmup_steps"
    }

# Network
    $lr_scheduler = $lr_scheduler.ToLower()

# Performance
    if ($full_fp16 -eq $true) {
        $extra += "--full_fp16"
    }
    if ($grad_checkpt -eq $true) {
        $extra += "--gradient_checkpointing"
    }
    if ($fp8_base -eq $true) {
        $extra += "--fp8_base"
    }
    if ($latents_disk -eq $true) {
        $extra += "--cache_latents_to_disk"
    }

# Debugging
    if ($correct_alpha -eq $true) {
        $net_alpha *= [Math]::Sqrt($net_dim)
    }
    if ($scale_lr -eq $true) {
        $unet_lr *= $total_grad_acc
        $text_enc_lr *= $total_grad_acc
    }
    if (($optimizer -eq "prodigy") -or ($optimizer -eq "dadaptadam")) {
        $is_lr_free = $true
    }
    if ($is_lr_free -eq $true) {
        $unet_lr = 1
        $text_enc_lr = 1
        $weight_decay = 0.0
        $opt_args += "decouple=True","use_bias_correction=True"
        $extra += "--max_grad_norm=0"
        if ($optimizer -eq "prodigy") {
            $opt_args += "d_coef=1.0"
        }
    }
    $opt_args += "betas=0.9,0.99","weight_decay=$weight_decay"

# Advanced
    if ($cap_dropout -gt 0) {
    $extra += "--caption_dropout=$cap_dropout"
    }
    if ($net_dropout -gt 0) {
        $extra += "--network_dropout=$net_dropout"
    }
    if ($scale_weight -gt 0) {
        $extra += "--scale_weight_norms=$scale_weight"
    }

# Additional lines to automatically generate named folders:
$full_name =  $lora_name + "_" + $version
[String]$unique_output = "$output_dir\$full_name"

.\venv\scripts\activate

accelerate launch --num_cpu_threads_per_process 8 $run_script `
    --logging_dir="logs" --log_prefix="$lora_name" `
    --network_module="networks.lora" `
    --max_data_loader_n_workers=1 --persistent_data_loader_workers `
    --caption_extension=".txt" --shuffle_caption --keep_tokens="$keep_tags" --max_token_length=225 `
    --prior_loss_weight=1 `
    --mixed_precision="fp16" --save_precision="fp16" `
    --xformers --cache_latents --save_model_as=safetensors `
    --train_data_dir="$dataset_dir" --output_dir="$unique_output" --reg_data_dir="$reg_dir" --pretrained_model_name_or_path="$checkpoint_dir_full" `
    --output_name="$full_name" `
    --unet_lr="$unet_lr" --text_encoder_lr="$text_enc_lr" `
    --resolution="$base_res" --enable_bucket --min_bucket_reso="$min_bucket_res" --max_bucket_reso="$max_bucket_res" --bucket_reso_steps="$bucket_step" `
    --train_batch_size="$batch_size" --gradient_accumulation_steps="$grad_acc_step" `
    --optimizer_type="$optimiser" --lr_scheduler="$lr_scheduler" --network_dim="$net_dim" --network_alpha="$net_alpha" `
    --noise_offset="$noise_offset" `
    --seed="$seed" `
    --clip_skip="$clip_skip" `
    --max_train_steps="$total_steps" --save_every_n_steps="$save_nth_step" `
    --min_snr_gamma=5 `
    --optimizer_args $opt_args `
    $extra `

Write-Host "Training finished!" -ForegroundColor green
pause
deactivate

# Sources:
# [1] == https://math.stackexchange.com/questions/2133509/how-do-i-calculate-the-length-and-width-for-a-known-area-and-ratio

# This script is originally derived from the powershell script in Raven's LoRA Training Rentry: https://rentry.org/59xed3
# Please ask Furry Diffusion about your LoRA training queries before consulting the above Rentry; it is not endorsed by Furry Diffusion.