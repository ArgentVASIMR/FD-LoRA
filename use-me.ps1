# This script is intended for use with sd-scripts, please install that if you haven't already:
# https://github.com/kohya-ss/sd-scripts

# Please visit the Furry Diffusion LoRA repo!
# https://github.com/ArgentVASIMR/FD-lora

# Main contributors (discord): 
    # argentvasimr
    # feffy
    # mmmmmmmm
# Last edited 8-03-2024 (argentvasimr)

# IF THE DATE ABOVE IS OLDER THAN A MONTH, PLEASE CHECK THE REPO FOR LATEST: https://github.com/ArgentVASIMR/FD-lora

# To implement:
    # Space-tolerant directories
    # Fix unicode issue (if fixable)

# For assistance, please ask in the LoRA training thread on Furry Diffusion: https://discord.com/channels/1019133813105905664/1065749961083469884

# Directories:
    $dataset_dir = ".\.dataset"
    $output_dir = ".\.output"
    $class_dir = ".\.class_img"
    $base_model_dir = "C:\your\models\folder\here"
    $prompts = ""; # Leave blank if you don't want sample images

# Base Model Config:
    $base_model     = "v1-5-pruned-emaonly.safetensors"
    $clip_skip      = 1
    $v_prediction   = $false
    $sdxl           = $false

# Training Config:
    # Output Settings:
        $lora_name      = "myFirstLoRA"
        $version        = "prototype"
        $comment        = "This is my first LoRA"
        $save_amount    = 10 # 0 to disable

    # Dataset Treatment:
        $base_res       = 512
        $bucket_step    = 64
        $flip_aug       = $true
        $keep_tags      = 1

    # Steps:
        $base_steps     = 2000
        $batch_size     = 1
        $grad_acc_step  = 1
        $warmup         = 0.1 # 0 to disable
        $warmup_type    = "percent" # "percent", "steps", "steps_batch"

    # Learning Rate:
        $unet_lr        = 1e-4
        $text_enc_lr    = 5e-5
        $scale_lr_batch = $true # Scale learning rate by batch size.
        $lr_scheduler   = "cosine" # Recommended options are "cosine", "linear".

    # Network:
        $net_dim        = 16
        $net_alpha      = 16
        $optimiser      = "adamw" # Recommended options are "adamw", "adamw8bit", "dadaptadam".
        $correct_alpha  = $false # Apply scaling to alpha, multiplying by sqrt($net_dim)

    # Performance:
        $grad_checkpt   = $false
        $full_precision = $false
        $fp8_base       = $false
        $latents_disk   = $false

# =============================================================================================
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# =============================================================================================

    # Debugging:
        $warnings       = $true
        $is_lr_free     = $false
        $max_aspect     = 2
        $precision      = "fp16"

    # Advanced:
    # (MAY BE REMOVED FROM CONFIG OUTRIGHT IN FUTURE IF NO ADDITIONAL CHANGES ARE RECOMMENDED)
        $weight_decay   = 0.01
        $seed           = 0
        $cap_dropout    = 0.0
        $net_dropout    = 0.1
        $scale_weight   = 0
        $d_coef         = 1.0

# Declaring/setting variables for later
    $run_script = "train_network.py"
    $extra = @()
    $opt_args = @()

# Directories
    if ($prompts -ne "") {
        $extra += "--sample_prompts=$prompts", "--sample_sampler=euler", "--sample_every_n_steps=$save_nth_step"
    }

    $full_name =  $lora_name + "_" + $version
    [String]$unique_output = "$output_dir\$full_name"

# Model Settings
    $base_model_dir_full = "$base_model_dir\$base_model"
    
    if ($v_prediction -eq $true) {
        $extra += "--v_parameterization", "--zero_terminal_snr"
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


$generic_warning = "If you do not want warnings, set `$warnings to false."

# Dataset Treatment
    if (($base_res -lt 512) -and ($warnings -eq $true)){
        Write-Host "WARNING: Your base resolution is set to less than 512 pixels, which is lower than what SD 1.5 is trained at." -ForegroundColor Yellow
        Write-Host "If this is intentional, then proceed by pressing enter. Otherwise, close this window to fix your settings." -ForegroundColor Yellow
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
    } elseif (($base_res -lt 1024) -and ($sdxl -eq $true) -and ($warnings -eq $true)){
        Write-Host "WARNING: Your base resolution is set to less than 1024 pixels, which is lower than what SDXL is trained at." -ForegroundColor Yellow
        Write-Host "If this is intentional, then proceed by pressing enter. Otherwise, close this window to fix your settings." -ForegroundColor Yellow
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
    }

    if ((($bucket_step/8) -isnot [int]) -and ($warnings -eq $true)){
        Write-Host "ERROR: `$bucket_step must be a multiple of 8 when using SD 1.5 models or similar." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    } elseif ((($bucket_step/32) -isnot [int]) -and ($sdxl -eq $true)  -and ($warnings -eq $true)){
        Write-Host "ERROR: `$bucket_step must be a multiple of 32 when using SDXL models." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }

    if ($flip_aug -eq $true) {
        $extra += "--flip_aug"
    }

# Steps
    $total_grad_acc = $batch_size*$grad_acc_step
    $base_steps = [int]([Math]::Round($base_steps / $total_grad_acc))
    if ($save_amount -gt 0){
        $save_nth_step = [int]([Math]::Round($base_steps / $save_amount))
        $extra += "--save_every_n_steps=$save_nth_step"
    }

# Check if warmup is set to a valid value with the warmup type
    if (($warmup -ge 1) -and ($warmup_type -eq "percent") -and ($warnings -eq $true)) {
        Write-Host "ERROR: The warmup percentage is set to equal to or greater than 1.0 ($warmup)." -ForegroundColor Red
        Write-Host "Please set `$warmup to less than 1 (recommended value is 0.1), OR set `$warmup_type to `"steps`" or `"steps_batch`"." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    } elseif (($warmup -lt 1) -and ($warmup_type -ne "percent") -and ($warnings -eq $true)) {
        Write-Host "ERROR: The warmup steps are less than 1 ($warmup)." -ForegroundColor Red
        Write-Host "Please set `$warmup to an integer greater than 1 (recommended value is 200), OR set `$warmup_type to `"percent`"." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    } elseif (($warmup -isnot [int]) -and ($warmup_type -ne "percent") -and ($warnings -eq $true)) {
        Write-Host "ERROR: The inputted warmup steps is a decimal and not an integer." -ForegroundColor Red
        Write-Host "Please set an integer value for `$warmup or set `$warmup_type to `"percent`"." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }

    if (($warmup_type -eq "percent") -or ($warmup_type -eq "steps_batch")){
        $warmup_steps = [int]([Math]::Round(($base_steps*$warmup)))
    } elseif ($warmup_type -eq "steps") {
        $warmup_steps = $warmup
    }
    if ($warmup_steps -gt $base_steps) {
        Write-Host "ERROR: Warmup steps cannot be greater than your base steps." -ForegroundColor Red
        Write-Host "Please ensure `$warmup is set to a number less than your base steps." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }
    if ($warmup -gt 0) {
        $extra += "--lr_warmup_steps=$warmup_steps"
    }

    Write-Host "Past!"

# Network
    $lr_scheduler = $lr_scheduler.ToLower()

# Performance
    if ($grad_checkpt -eq $true) {
        $extra += "--gradient_checkpointing"
    }
    if (($full_precision -eq $true) -and ($precision = "fp16")) {
        $extra += "--full_fp16"
    }
    if (($full_precision -eq $true) -and ($precision = "bf16")) {
        $extra += "--full_bf16"
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
    if ($scale_lr_batch -eq $true) {
        $unet_lr *= $total_grad_acc
        $text_enc_lr *= $total_grad_acc
    }
    if (($optimiser -eq "prodigy") -or ($optimiser -eq "dadaptadam") -or ($optimiser -eq "dadaptation")) {
        $is_lr_free = $true
    }
    if ($is_lr_free -eq $true) {
        $unet_lr = 1
        $text_enc_lr = 1
        $opt_args += "decouple=True","use_bias_correction=True"
        $extra += "--max_grad_norm=0"
        if ($optimiser -eq "prodigy") {
            $opt_args += "d_coef=$d_coef"
        }
    }
    $opt_args += "betas=0.9,0.99", "weight_decay=$weight_decay"

    # Bucketing res calculations via $max_aspect [1]
    if ($max_aspect -lt 1) {
        $max_aspect = 1/$max_aspect # Flip aspect ratio if it's less than 1
    }
    $max_bucket_res = [int]([Math]::Ceiling([Math]::Sqrt(($base_res * $base_res * $max_aspect)) / 64) * 64)
    $min_bucket_res = [int]([Math]::Floor([Math]::Sqrt(($base_res * $base_res / $max_aspect)) / 64) * 64)

# Advanced
    if ($cap_dropout -gt 0) {
    $extra += "--caption_dropout_rate=$cap_dropout"
    }
    if ($net_dropout -gt 0) {
        $extra += "--network_dropout=$net_dropout"
    }
    if ($scale_weight -gt 0) {
        $extra += "--scale_weight_norms=$scale_weight"
    }

.\venv\scripts\activate

accelerate launch --num_cpu_threads_per_process 8 $run_script `
    --logging_dir="logs" --log_prefix="$lora_name" `
    --network_module="networks.lora" `
    --max_data_loader_n_workers=1 --persistent_data_loader_workers `
    --caption_extension=".txt" --shuffle_caption --keep_tokens="$keep_tags" --max_token_length=225 `
    --prior_loss_weight=1 `
    --mixed_precision="$precision" --save_precision="$precision" `
    --xformers --cache_latents --save_model_as=safetensors `
    --train_data_dir="$dataset_dir" --output_dir="$unique_output" --reg_data_dir="$class_dir" --pretrained_model_name_or_path="$base_model_dir_full" `
    --output_name="$full_name" `
    --unet_lr="$unet_lr" --text_encoder_lr="$text_enc_lr" `
    --resolution="$base_res" --enable_bucket --min_bucket_reso="$min_bucket_res" --max_bucket_reso="$max_bucket_res" --bucket_reso_steps="$bucket_step" `
    --train_batch_size="$batch_size" --gradient_accumulation_steps="$grad_acc_step" `
    --optimizer_type="$optimiser" --lr_scheduler="$lr_scheduler" --network_dim="$net_dim" --network_alpha="$net_alpha" `
    --noise_offset="$noise_offset" `
    --seed="$seed" `
    --clip_skip="$clip_skip" `
    --max_train_steps="$base_steps" `
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