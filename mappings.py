renames = {
    "log_dir": "logging_dir",
    "keep_tags": "keep_tokens",
    "dataset_dir": "train_data_dir",
    "unique_output": "output_dir",
    "class_dir": "reg_data_dir",
    "base_model_dir_full": "pretrained_model_name_or_path",
    "full_name": "output_name",
    'text_enc_lr': 'text_encoder_lr',
    'base_res': 'resolution',
    'bucket_step': 'bucket_reso_steps',
    'batch_size': 'train_batch_size',
    'grad_acc_step': 'gradient_accumulation_steps',
    'optimiser': 'optimizer_type',
    'net_dim': 'network_dim',
    'net_alpha': 'network_alpha',
    'base_steps': 'max_train_steps',
    'opt_args': 'optimizer_args',
    'grad_checkpt': 'gradient_checkpointing',
    'cap_dropout': 'caption_dropout_rate',
    'net_dropout': 'network_dropout',
    'scale_weight': 'scale_weight_norms',
    'tag_dropout': 'caption_tag_dropout_rate',
    'precision': 'mixed_precision',
    'warmup_steps': 'lr_warmup_steps'
}
constants = {
    "network_module": "networks.lora",
    "max_data_loader_n_workers": 1,
    "persistent_data_loader_workers": True,
    "caption_extension": ".txt",
    "prior_loss_weight": 1,
    "max_token_length": 225,
    "xformers": True,
    "cache_latents": True,
    'enable_bucket': True,
}
remove_key_list = ['sdxl','base_model','base_model_dir','version','lora_name',
               'scale_steps','scale_lr','save_amount','warnings','deactivate','notify','pause_at_end','class_2x_steps','warmup_always','old_lr_scale',
               'warmup','warmup_type','unet_only', 'lora_weight', 
               'resume', 'old_version', 'resume_path']

optimizer_args = ['weight_decay','d_coef']
#TODO implement:
    #warnings
    #deactivate
    #notify
    #pause_at_end
    #class_2x_steps
    #warmup_always
    #old_lr_scale
    #save_amount
    #scale_steps
    #unet_only (enum)
greaters = ['cap_dropout', 'net_dropout', 'scale_weight', 'tag_dropout']


def rename_keys(config : dict):
    for k, v in renames.items():
        if k in config:
            config[v] = config.pop(k)
def remove_keys(config : dict):
    [config.pop(k) for k in remove_key_list if k in config]

def optimizer_arg_mapping(config : dict):
    list = []
    for k in optimizer_args:
        if k in config:
            if k == 'd_coef' and config['optimiser'] != 'prodigy': continue
            list.append(f"{k}={config.pop(k)}")
    config['optimizer_args'] = list

def other_mappings(config : dict):
    config['save_precision'] = config['precision']
    config['log_prefix'] = config['lora_name']

def check_minimums(config : dict): # TODO add warnings
    if config['base_res'] < 512 and config['sdxl'] == False: 
        config['base_res'] = 512
    elif config['base_res'] < 1024 and config['sdxl'] == True:
        config['base_res'] = 1024

    if config['bucket_step']%32 != 0 and config['sdxl'] == True:
        config['bucket_step'] = config['bucket_step'] - (config['bucket_step']%32)
    elif config['bucket_step']%8 != 0 and config['sdxl'] == False:
        config['bucket_step'] = config['bucket_step'] - (config['bucket_step']%8)

    if config['bucket_step'] > config['base_res']:
        config['bucket_step'] = config['base_res'] #you really need to warn about this one going forwards
    
    for k in greaters:
        if config[k] == 0: config.pop(k)
def scale_lr(config : dict):
    eff_batch_size = config['batch_size'] * config['grad_acc_step']
    if config['scale_lr'] and eff_batch_size > 1:
        config['unet_lr'] *= eff_batch_size ** 0.5
        config['text_enc_lr'] *= eff_batch_size ** 0.5

def warmup_steps(config : dict): #TODO warnings
    if config['warmup'] == 0: return
    #warmup types: percent, steps, steps_batch
    #if warmup is percent, it is a float between 0 and 1. otherwise it is an int
    if config['warmup_type'] == 'percent' or config['warmup_type'] == 'steps_batch':
        config['warmup_steps'] = int(config['base_steps'] * config['warmup'])
    elif config['warmup_type'] == 'steps':
        config['warmup_steps'] = config['warmup']
    else:
        #TODO warn that this is not a valid warmup type
        pass
    config.pop('warmup')
    config.pop('warmup_type')

def preprocess_config(config : dict): 
    config.update(constants)
    #directories
    config['base_model_dir_full'] = config['base_model_dir'] + '\\' + config['base_model']
    config['full_name'] = f"{config['lora_name']}_{config['version']}"
    config['unique_output'] = config['output_dir'] + '\\' + config['full_name']
    
    #lora weight calculations
    if config['precision'] == 'auto':
        if config['lora_weight'] == 'fp32': config['precision'] = 'fp16'
        else: config['precision'] = config['lora_weight']
    if config['lora_weight'] == 'fp16': config['full_fp16'] = True
    if config['lora_weight'] == 'bf16': config['full_bf16'] = True
    
    scale_lr(config)
    warmup_steps(config)

    check_minimums(config)
    other_mappings(config)
    rename_keys(config)
    optimizer_arg_mapping(config)
    remove_keys(config)

    return config
