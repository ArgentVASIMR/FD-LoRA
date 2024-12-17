from logger import Logger
from validator import Validator
import os
renames={
    "keep_tags": "keep_tokens",
    "class_dir": "reg_data_dir", #TODO see if this is actually correct nomenclature
    "dataset_dir": "train_data_dir",
    "base_steps": "max_train_steps",
    "tag_dropout": "caption_tag_dropout_rate",
    "precision": "mixed_precision",
    "base_res": "resolution",
}
constants = {
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
               'resume', 'old_version', 'resume_path','handle_errors']

#TODO implement:
    #warnings
    #deactivate
    #notify
    #pause_at_end
    #verification python class
    #warmup_always
    #unet_only (enum)
    #lycoris/DoRA support
greaters = ['caption_dropout_rate', 'network_dropout', 'scale_weight_norms', 'tag_dropout']

def check_minimums(config : dict): # TODO add warnings
    for k in greaters:
        if config[k] == 0: config.pop(k)
class Mapper:
    def __init__(self,logger: Logger,config : dict):
        self.config = config
        self.logger = logger
    def rename_keys(self):
        config = self.config
        modified_list = []
        for k, v in renames.items():
            if k in config:
                config[v] = config.pop(k)
                modified_list.append(k)
        self.logger.debug(f"Renamed {modified_list}")
    def remove_keys(self):
        [self.config.pop(k) for k in remove_key_list if k in self.config]
        self.logger.debug(f"Removed {remove_key_list}")
    def optimizer_arg_mapping(self):
        config = self.config
        lis = []
        for k, val in config['optimizer_args'].items():
            if k == 'd_coef':
                if config['optimizer_type'] == 'prodigy': 
                    self.logger.info(f"prodigy detected, adding dcoef={val}")
                else: continue
            lis.append(f"{k}={val}")
        self.logger.debug(f"Added {lis} to optimizer_args")
        config['optimizer_args'] = lis
    def other_mappings(self):
        config = self.config
        config['save_precision'] = config['precision']
        config['log_prefix'] = config['lora_name']
        config['max_bucket_reso'] = config['base_res']*2
    def scale_lr(self):
        config = self.config
        eff_batch_size = config['train_batch_size'] * config['gradient_accumulation_steps']
        if config['scale_lr'] and eff_batch_size > 1:
            old_unet_lr = config['unet_lr']
            old_te_lr = config['text_encoder_lr']

            config['unet_lr'] *= eff_batch_size ** 0.5
            config['text_encoder_lr'] *= eff_batch_size ** 0.5

            self.logger.info(f"scale_lr is set to true, learning rates have been adjusted to compensate:")
            self.logger.info(f"Unet LR: {old_unet_lr} --> {config['unet_lr']}")
            self.logger.info(f"Text Encoder LR: {old_te_lr} --> {config['text_encoder_lr']}")
    def scale_steps(self):
        config = self.config
        eff_batch_size = config['train_batch_size'] * config['gradient_accumulation_steps']
        if config['scale_steps'] and eff_batch_size > 1:
            old_steps = config['base_steps']
            config['base_steps'] = int(config['base_steps'] / eff_batch_size)
            self.logger.info(f"scale_steps is set to true and batch size is {eff_batch_size}, step count has been adjusted to compensate:")
            self.logger.info(f"Step count: {old_steps} --> {config['base_steps']}")
    
    def warmup_steps(self):
        config = self.config
        if config['warmup'] == 0: 
            self.logger.info("Warmup is set to 0, skipping warmup steps")
            return
        #warmup types: percent, steps, steps_batch
        #if warmup is percent, it is a float between 0 and 1. otherwise it is an int
        if config['warmup_type'] == 'percent' or config['warmup_type'] == 'steps_batch':
            config['lr_warmup_steps'] = int(config['base_steps'] * config['warmup'])
        elif config['warmup_type'] == 'steps':
            config['lr_warmup_steps'] = config['warmup']
        else:
            self.error(f"Invalid warmup type: {config['warmup_type']}, must be percent, steps, or steps_batch")
        self.logger.info(f"Warmup steps: {config['lr_warmup_steps']}")
        config.pop('warmup')
        config.pop('warmup_type')
    
    def save_config(self):
        config = self.config
        if(config['save_amount'] > 0):
            config['save_every_n_steps'] = int(config['base_steps'] / config['save_amount'])
            self.logger.info(f"save_every_n_steps set to {config['save_every_n_steps']}")
    def preprocess_config(self): 
        config = self.config
        config.update(constants)
        #directories
        config['pretrained_model_name_or_path'] = os.path.join(config['base_model_dir'], config['base_model'])
        name = f"{config['lora_name']}_{config['version']}"
        config['output_name'] = os.path.join(config['output_dir'], name)
        
        #lora weight calculations
        if config['precision'] == 'auto':
            if config['lora_weight'] == 'fp32': config['precision'] = 'fp16'
            else: config['precision'] = config['lora_weight']
        if config['lora_weight'] == 'fp16': config['full_fp16'] = True
        if config['lora_weight'] == 'bf16': config['full_bf16'] = True
        

        self.scale_lr()
        self.scale_steps()
        self.warmup_steps()
        self.save_config()

        check_minimums(config) #TODO make this into a class method
        self.other_mappings()
        self.rename_keys()
        self.optimizer_arg_mapping()
        self.remove_keys()
