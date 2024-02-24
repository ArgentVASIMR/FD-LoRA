# This script is intended for use with sd-scripts, please install that if you haven't already:
# https://github.com/kohya-ss/sd-scripts

# Please visit the Furry Diffusion LoRA repo!
# https://github.com/ArgentVASIMR/FD-lora

# Main contributors (discord): 
    # argentvasimr
    # feffy
    # mmmmmmmm
# Last edited 24-02-2024 (argentvasimr)
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
# Basic Settings:
    $lora_name      = "test"
    $version        = "prototype"
    $save_amount    = 10
    $base_res       = 768
    $max_aspect     = 1.5
    $bucket_step    = 8
    $flip_aug       = $true
    $keep_tags      = 1

# Steps:
    $total_steps    = 2000
    $batch_size     = 16
    $grad_acc_step  = 1
    $warmup         = 200
    $warmup_type    = "steps" # "steps", "steps_batch", "percent"

# Learning Rate:
    $unet_lr        = 1e-4
    $text_enc_lr    = 5e-5
    $lr_scheduler   = "cosine" # Recommended options are "cosine", "linear".

# Network:
    $net_dim        = 32
    $net_alpha      = 32
    $optimiser      = "adamw" # Recommended options are "adamw", "adamw8bit", "dadaptadam".

# Debugging:
    $correct_alpha  = $false # Apply scaling to alpha, multiplying by sqrt($net_dim)
    $scale_lr       = $true # Scale learning rate by batch size.
    $is_lr_free     = $false # Manual toggle for LR-free optimisers, in case one isn't accounted for.

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

# Almost-Never-Touched Settings:
# (MAY BE REMOVED FROM CONFIG OUTRIGHT IN FUTURE IF NO ADDITIONAL CHANGES ARE RECOMMENDED)
    $weight_decay   = 0.01
    $seed           = 0
    $cap_dropout    = 0
    $net_dropout    = 0.1
    $scale_weight   = 1

if (($warmup -ge 1) -and ($warmup_type -eq "percent")) {
Write-Host "The warmup percentage is set to equal to or greater than 1.0 / 100% ($warmup). Please set `$warmup to less than 1 to work (recommended value is 0.1 / 10%)"
pause
exit
}

# Adaptive optimiser settings
if (($optimizer -eq "prodigy") -or ($optimizer -eq "dadaptadam")) {
$is_lr_free = $true
}

if ($is_lr_free -eq $true) {
$unet_lr = 1
$text_enc_lr = 1
$weight_decay = 0.0
}

if ($correct_alpha -eq $true) {
$net_alpha *= [Math]::Sqrt($net_dim)
}

$run_script = "train_network.py"
if ($sdxl -eq $true) {
$run_script = "SDXL_train_network.py"
}

$lr_scheduler = $lr_scheduler.ToLower()

$total_grad_acc = $batch_size*$grad_acc_step

if ($scale_lr -eq $true) {
$unet_lr *= $total_grad_acc
$text_enc_lr *= $total_grad_acc
}

$total_steps = [int]([Math]::Round($total_steps / $total_grad_acc))
$save_nth_step = [int]([Math]::Round($total_steps / $save_amount))

if (($warmup_type -eq "percent") -or ($warmup_type -eq "steps_batch")){
$warmup_steps = [int]([Math]::Round(($total_steps*$warmup)))
} elseif ($warmup_type -eq "steps") {
$warmup_steps = $warmup
}

# Add optional args
$extra = @()

if ($prompts -ne "") {
$extra += "--sample_prompts=$prompts","--sample_sampler=euler","--sample_every_n_steps=$save_nth_step"
}
if ($v_prediction -eq $true) {
$extra += "--v_parameterization","--zero_terminal_snr"
$noise_offset = 0.0
} else {
$noise_offset = 0.01
}
if ($flip_aug -eq $true) {
$extra += "--flip_aug"
}
if ($scale_weight -gt 0) {
$extra += "--scale_weight_norms=$scale_weight"
}
if ($min_snr_gamma -gt 0) {
$extra += "--min_snr_gamma=5"
}
if ($latents_disk -eq $true) {
$extra += "--cache_latents_to_disk"
}
if ($cap_dropout -gt 0) {
$extra += "--caption_dropout=$cap_dropout"
}
if ($net_dropout -gt 0) {
$extra += "--network_dropout=$net_dropout"
}
if ($warmup_steps -gt 0) {
$extra += "--lr_warmup_steps=$warmup_steps"
}

# Performance Settings
if ($full_fp16 -eq $true) {
$extra += "--full_fp16"
}
if ($grad_checkpt -eq $true) {
$extra += "--gradient_checkpointing"
}
if ($fp8_base -eq $true) {
$extra += "--fp8_base"
}



$optArgs = "betas=0.9,0.99","weight_decay=$weight_decay"

if ($is_adaptive -eq $true) {
$optArgs += "decouple=True","use_bias_correction=True"
[void]$extra.Add("--max_grad_norm=0")
if ($optimizer -eq "prodigy") {
    $optArgs += "d_coef=1.0"
}
}

# Bucketing res calculation [1]
if ($max_aspect -lt 1) {$max_aspect = 1/$max_aspect} # Flip aspect ratio if it's less than 1
$max_bucket_res = [int]([Math]::Ceiling([Math]::Sqrt(($base_res * $base_res * $max_aspect)) / 64) * 64)
$min_bucket_res = [int]([Math]::Floor([Math]::Sqrt(($base_res * $base_res / $max_aspect)) / 64) * 64)

if ($max_bucket_res -lt 8) {
$max_bucket_res = 8
}
if ($min_bucket_res -lt 8) {
$min_bucket_res = 8
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
--train_data_dir="$dataset_dir" --output_dir="$unique_output" --reg_data_dir="$reg_dir" --pretrained_model_name_or_path="$checkpoint_dir\$checkpoint" `
--output_name="$full_name" `
--learning_rate="$unet_lr" --text_encoder_lr="$text_enc_lr" `
--resolution="$base_res" --enable_bucket --min_bucket_reso="$min_bucket_res" --max_bucket_reso="$max_bucket_res" --bucket_reso_steps="$bucket_step" `
--train_batch_size="$batch_size" --gradient_accumulation_steps="$grad_acc_step" `
--optimizer_type="$optimiser" --lr_scheduler="$lr_scheduler" --network_dim="$net_dim" --network_alpha="$net_alpha" `
--noise_offset="$noise_offset" `
--seed="$seed" `
--clip_skip="$clip_skip" `
--max_train_steps="$total_steps" --save_every_n_steps="$save_nth_step" `
--optimizer_args $optArgs `
$extra `

pause

# Sources:
# [1] == https://math.stackexchange.com/questions/2133509/how-do-i-calculate-the-length-and-width-for-a-known-area-and-ratio

# This script is originally derived from the powershell script in Raven's LoRA Training Rentry: https://rentry.org/59xed3
# Please ask Furry Diffusion about your LoRA training queries before consulting the above Rentry; it is not endorsed by Furry Diffusion.