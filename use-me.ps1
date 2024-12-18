# This script is intended for use with sd-scripts, please install that if you haven't already:
# https://github.com/ArgentVASIMR/FD-lora/wiki/Installing-sd%E2%80%90scripts-(Our-Method,-For-Windows)

# Please visit the Furry Diffusion LoRA repo!
# https://github.com/ArgentVASIMR/FD-lora

# Last edited 15-11-2024 (D/M/Y) (argentvasimr)

# IF THE DATE ABOVE IS OLDER THAN A MONTH, PLEASE CHECK THE REPO FOR LATEST: https://github.com/ArgentVASIMR/FD-lora

# To implement:
    # Space-tolerant directories
    # Automated tensorboard startup

# For assistance, please ask in either: 
# - LoRA support forum on Furry Diffusion: https://discord.com/channels/1019133813105905664/1216117427700760586
# - Github repo thread on Furry Diffusion: https://discord.com/channels/1019133813105905664/1213336377563815987
# - Submit an issue in the repo itself: https://github.com/ArgentVASIMR/FD-LoRA/issues

# Directories:
    $dataset_dir = ".\.dataset"
    $output_dir = ".\.output"
    $class_dir = ".\.class_img"
    $base_model_dir = "C:\your\models\folder\here"
    $prompts = ""; # Leave blank if you don't want sample images
    $log_dir = "logs";

# Base Model Config:
    $base_model     = "v1-5-pruned-emaonly.safetensors"
    $clip_skip      = 1
    $v_prediction   = $false
    $sdxl           = $false

# Training Config:
    # Output Settings:
        $lora_name     = "myFirstLoRA"
        $version       = "prototype"
        $comment       = "This is my first LoRA"
        $save_amount   = 10 # 0 to disable (doing so will also disable image previews)

    # Dataset Treatment:
        $base_res      = 512
        $bucket_step   = 64
        $flip_aug      = $true
        $keep_tags     = 1

    # Steps:
        $base_steps    = 2000
        $batch_size    = 1
        $grad_acc_step = 1
        $scale_steps   = $true
        $warmup        = 0.1 # 0 to disable
        $warmup_type   = "percent" # Options are "percent", "steps", "steps_batch"

    # Learning Rate:
        $unet_lr       = 1e-4
        $text_enc_lr   = 5e-5
        $scale_lr      = $true
        $lr_scheduler  = "cosine" # Options are "cosine", "linear", "constant", "constant_with_warmup".

    # Network:
        $net_dim       = 16
        $net_alpha     = 16
        $optimiser     = "adamw" # Options are "adamw", "adamw8bit", "dadaptadam".

    # Performance:
        $grad_checkpt  = $false
        $lora_weight   = "fp32" # Options are "fp32", "fp16", "bf16"
        $fp8_base      = $false
        $unet_only     = 0 # 0 is off, 1 is partial, 2 is fully
        $mem_eff_attn  = $false
        $xformers      = $true
        $sdpa          = $false

# =============================================================================================
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# [!!!] BEYOND THIS POINT IS STUFF YOU SHOULD NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING [!!!]
# =============================================================================================

    # Debugging:
        $warnings       = $true
        $is_lr_free     = $false    
        $pause_at_end   = $true
        $deactivate     = $true
        $precision      = "auto" # Options are "auto", "fp16", "bf16"
        $class_2x_steps = $true
        $warmup_always  = $false
        $old_lr_scale   = $false

    # Advanced:
    # (MAY BE REMOVED FROM CONFIG IN FUTURE IF NO ADDITIONAL CHANGES ARE RECOMMENDED)
        $weight_decay   = 0.01
        $seed           = 1
        $cap_dropout    = 0.0
        $net_dropout    = 0.1
        $tag_dropout    = 0.0
        $scale_weight   = 0
        $d_coef         = 1.0
        $noise_offset   = 0.0375
        $min_snr_gamma  = 1
        $max_grad_norm  = 1
        $correct_alpha  = $false # Apply scaling to alpha, multiplying by sqrt($net_dim)
        $loss_type      = "l2" # Options are "l2", "huber"
        $maxed_buckets  = $true

        $extra = @() # Add args to here instead of editing the args at the bottom of this script
        $opt_args = @() # Add args to here instead of editing the args at the bottom of this script

# Declaring/setting variables for later
    $run_script = "train_network.py"
    $generic_warning = "If you do not want warnings, set `$warnings to false."
    $generic_optional = "If this is intentional, then proceed by pressing enter. Otherwise, shut down this process and fix your settings."
    $eff_batch_size = $batch_size*$grad_acc_step

    # Multiplying learning rates by batch size / gradient accumulation steps
    if (($scale_lr -eq $true) -and ($eff_batch_size -gt 1)){
        $old_unet_lr = $unet_lr
        $old_te_lr = $text_enc_lr

        if ($old_lr_scale -eq $true){
            $unet_lr *= $eff_batch_size
            $text_enc_lr *= $eff_batch_size
        } else {
            $unet_lr *= [Math]::Sqrt($eff_batch_size)
            $text_enc_lr *= [Math]::Sqrt($eff_batch_size)
        }

        Write-Host "`$scale_lr is set to true, learning rates have been adjusted to compensate:"
        Write-Host "Unet LR: $old_unet_lr --> $unet_lr"
        Write-Host "Text Encoder LR: $old_te_lr --> $text_enc_lr"
    }

    # Auto-adjusting precision based on LoRA weight precision setting:
    if ($precision -eq "auto"){
        if ($lora_weight -eq "fp32"){
            $precision = "fp16"
        } else {
            $precision = $lora_weight
        }
    }

# Directories
    $full_name =  $lora_name + "_" + $version
    [String]$unique_output = "$output_dir\$full_name"

# Messages to inform the user on whats going on
    if ($class_2x_steps -eq $true) {
        $classdata = Get-ChildItem -Path $class_dir

        if ($classdata.Count -gt 0) {
            Write-Host "Class dataset found; doubling step count."
            $base_steps *= 2
            $warmup_steps *= 2
            $save_amount /= 2
        }
    }
    
# Model Settings
    $base_model_dir_full = "$base_model_dir\$base_model"
    
    if ($v_prediction -eq $true){
        $extra += "--v_parameterization", "--zero_terminal_snr"
        $noise_offset = 0.0
    }
    if ($sdxl -eq $true){
        $run_script = "SDXL_train_network.py"
    }

# Output Settings
    if ($comment -ne ""){
        $extra += "--training_comment=$comment"
    }

# Dataset Treatment
    if (($base_res -lt 512) -and ($warnings -eq $true)){
        Write-Host "WARNING: Your base resolution is set to less than 512 pixels, which is lower than what SD 1.5 is trained at." -ForegroundColor Yellow
        Write-Host "$generic_optional" -ForegroundColor Yellow
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
    } elseif (($base_res -lt 1024) -and ($sdxl -eq $true) -and ($warnings -eq $true)){
        Write-Host "WARNING: Your base resolution is set to less than 1024 pixels, which is lower than what SDXL is trained at." -ForegroundColor Yellow
        Write-Host "$generic_optional" -ForegroundColor Yellow
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
    }

    if ((($bucket_step/8) -isnot [int]) -and ($warnings -eq $true)){
        Write-Host "ERROR: `$bucket_step must be a multiple of 8 when using SD 1.5 models or similar. It is currently set to $($bucket_step)." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    } elseif ((($bucket_step/32) -isnot [int]) -and ($sdxl -eq $true)  -and ($warnings -eq $true)){
        Write-Host "ERROR: `$bucket_step must be a multiple of 32 when using SDXL models. It is currently set to $($bucket_step)." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }
    if ($bucket_step -gt $base_res){
        Write-Host "ERROR: `$bucket_step is higher than `$base_res. If unsure of this setting, set it to 64 and forget about it." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }

    if ($flip_aug -eq $true){
        $extra += "--flip_aug"
    }

# Steps
    if (($scale_steps -eq $true) -and ($eff_batch_size -gt 1)){
        $old_steps = $base_steps
        $base_steps = [int]([Math]::Round($base_steps / $eff_batch_size))

        Write-Host "`$scale_lr_batch is set to true, step counts have been adjusted to compensate:"
        Write-Host "Step count: $old_steps --> $base_steps"
    }
    if ($save_amount -gt 0){
        $save_nth_step = [int]([Math]::Round($base_steps / $save_amount))
        $extra += "--save_every_n_steps=$save_nth_step"

        if ($prompts -ne "") {
            $extra += "--sample_prompts=$prompts", "--sample_sampler=euler", "--sample_every_n_steps=$save_nth_step"
        }
    }

# Check if warmup is set to a valid value with the warmup type
    if (($warmup -ge 1) -and ($warmup_type -eq "percent") -and ($warnings -eq $true)){
        Write-Host "ERROR: The warmup percentage is set to equal to or greater than 1.0 ($warmup)." -ForegroundColor Red
        Write-Host "Please set `$warmup to less than 1 (recommended value is 0.1), OR set `$warmup_type to `"steps`" or `"steps_batch`"." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    } elseif (($warmup -lt 1) -and ($warmup_type -ne "percent") -and ($warnings -eq $true)){
        Write-Host "ERROR: The warmup steps are less than 1 ($warmup)." -ForegroundColor Red
        Write-Host "Please set `$warmup to an integer greater than 1 (recommended value is 200), OR set `$warmup_type to `"percent`"." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    } elseif (($warmup -isnot [int]) -and ($warmup_type -ne "percent") -and ($warnings -eq $true)){
        Write-Host "ERROR: The inputted warmup steps is a decimal and not an integer." -ForegroundColor Red
        Write-Host "Please set an integer value for `$warmup or set `$warmup_type to `"percent`"." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }

    if (($warmup_type -eq "percent") -or ($warmup_type -eq "steps_batch")){
        $warmup_steps = [int]([Math]::Round(($base_steps*$warmup)))
    } elseif ($warmup_type -eq "steps"){
        $warmup_steps = $warmup
    }
    if ($warmup_steps -gt $base_steps){
        Write-Host "ERROR: Warmup steps cannot be greater than your base steps." -ForegroundColor Red
        Write-Host "Please ensure `$warmup is set to a number less than your base steps." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }

# Network
    $lr_scheduler = $lr_scheduler.ToLower()

# Performance
    if ($grad_checkpt -eq $true){
        $extra += "--gradient_checkpointing"
    }
    if ($lora_weight -eq "fp16"){
        $extra += "--full_fp16"
    }
    if ($lora_weight -eq "bf16"){
        $extra += "--full_bf16"
    }
    if ($fp8_base -eq $true){
        $extra += "--fp8_base"
    }
    if (($sdxl -eq $true) -and ($grad_checkpt -eq $false) -and ($warnings -eq $true)){
        Write-Host "WARNING: SDXL will use a lot of VRAM, and gradient checkpointing can help you avoid an `"Out of Memory`" error. You have gradient checkpointing turned off." -ForegroundColor Yellow
        Write-Host "If this is intentional, then proceed by pressing enter. Otherwise, close this window and fix your settings." -ForegroundColor Yellow
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
    }

    # Unet-only code
    if ($unet_only -eq (1 -or 2)){
        $extra += "--network_train_unet_only"
        if ($unet_only -eq 2){
            $extra += "--cache_text_encoder_outputs"
        }
    } elseif ($unet_only -eq 0){
        $extra += "--shuffle_caption"
    } else {
        Write-Host "ERROR: $unet_only set to an invalid value; please set it to 0, 1, or 2." -ForegroundColor Red
        Write-Host "$generic_warning" -ForegroundColor Magenta
        pause
        exit
    }

    # Additional optimisation args:
    if ($mem_eff_attn -eq $true){
        $extra += "--mem_eff_attn"
    }
    if ($xformers -eq $true){
        $extra += "--xformers"
    }
    if ($sdpa -eq $true){
        $extra += "--sdpa"
    }

# Debugging
    if ($correct_alpha -eq $true){
        $net_alpha *= [Math]::Sqrt($net_dim)
    }
    if (($optimiser -eq "prodigy") -or ($optimiser -eq "dadaptadam") -or ($optimiser -eq "dadaptation")){
        $is_lr_free = $true
    }
    if ($is_lr_free -eq $true){
        $unet_lr = 1
        $text_enc_lr = 1
        $opt_args += "decouple=True","use_bias_correction=True"
        $extra += "--max_grad_norm=$max_grad_norm"
        if ($optimiser -eq "prodigy"){
            $opt_args += "d_coef=$d_coef"
        }
    }
    if ($optimiser -ne "adafactor"){
        $opt_args += "betas=0.9,0.99"
    }
    $opt_args += "weight_decay=$weight_decay"

    if ((($optimiser -eq "prodigy") -or ($optimiser -eq "dadaptadam")) -and ($warmup_always -eq $false)){
        Write-Host "Optimiser is set to "$($optimiser)"; disabling warmup."
        $warmup = 0
    }
    if ($warmup -gt 0){
        $extra += "--lr_warmup_steps=$warmup_steps"
    }
    if (($warmup_always -eq $true) -and (($optimiser -eq "prodigy"))){
        Write-Host "Optimiser is set to "$($optimiser)" and `$warmup_always is set to `$true; enabling safeguard warmup."
        $extra += "--safeguard_warmup=True"
    }

# Advanced
    if ($cap_dropout -gt 0){
        $extra += "--caption_dropout_rate=$cap_dropout"
    }
    if ($net_dropout -gt 0){
        $extra += "--network_dropout=$net_dropout"
    }
    if ($scale_weight -gt 0){
        $extra += "--scale_weight_norms=$scale_weight"
    }
    if (($tag_dropout -gt 0) -and ($unet_only -eq (0 -or 1))){
        $extra += "--caption_tag_dropout_rate=$tag_dropout"
    }
    if ($maxed_buckets -eq $true){
        $extra += "--max_bucket_reso=$($base_res*2)"
    }

.\venv\scripts\activate

accelerate launch --num_cpu_threads_per_process 8 $run_script `
    --logging_dir="$log_dir" --log_prefix="$lora_name" `
    --network_module="networks.lora" `
    --max_data_loader_n_workers=1 --persistent_data_loader_workers `
    --caption_extension=".txt" --keep_tokens="$keep_tags" --max_token_length=225 `
    --prior_loss_weight=1 `
    --mixed_precision="$precision" --save_precision="$precision" `
    --cache_latents --save_model_as=safetensors `
    --train_data_dir="$dataset_dir" --output_dir="$unique_output" --reg_data_dir="$class_dir" --pretrained_model_name_or_path="$base_model_dir_full" `
    --output_name="$full_name" `
    --unet_lr="$unet_lr" --text_encoder_lr="$text_enc_lr" `
    --resolution="$base_res" --enable_bucket --bucket_reso_steps="$bucket_step" `
    --train_batch_size="$batch_size" --gradient_accumulation_steps="$grad_acc_step" `
    --optimizer_type="$optimiser" --lr_scheduler="$lr_scheduler" --network_dim="$net_dim" --network_alpha="$net_alpha" `
    --noise_offset="$noise_offset" `
    --seed="$seed" `
    --clip_skip="$clip_skip" `
    --max_train_steps="$base_steps" `
    --min_snr_gamma="$min_snr_gamma" `
    --optimizer_args $opt_args `
    --loss_type="$loss_type" `
    $extra `

if ($pause_at_end -eq $true){
    Write-Host "Process finished. If you don't want this pause, disable it by setting `$pause_at_end to false." -ForegroundColor green
    pause
}

if ($deactivate -eq $true){
    deactivate
}

# This script is originally derived from the powershell script in Raven's LoRA Training Rentry.
# Please ask Furry Diffusion about your LoRA training queries before consulting said Rentry; it is not endorsed by Furry Diffusion.