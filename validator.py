def to_nearest_multiple(n, m, raise_to_min=False):
    v = round(n / m) * m
    if v == 0 and raise_to_min:
        v = m
    return v

class Validator:
    def __init__(self, logger, config, handle_errors=False):
        self.logger = logger
        self.config = config
        self.handle_errors = handle_errors
    def error_if_unhandled(self, error_message,fix_message=""):
        if not self.handle_errors:
            self.logger.error(error_message)
        else:
            self.logger.warn(f"{error_message} -> {fix_message}")
    def validate_resolution(self):
        #TODO: warn instead of error for undersized resolutions
        if self.config['base_res'] < 512:
            self.error_if_unhandled("Base resolution must be at least 512","Setting base resolution to 512")
            self.config['base_res'] = 512
        if self.config['base_res'] < 1024 and self.config['sdxl'] == True:
            self.error_if_unhandled("Base resolution must be at least 1024 when using SDXL","Setting base resolution to 1024")
            self.config['base_res'] = 1024
    def validate_bucket_reso_steps(self):
        if self.config['bucket_reso_steps']%32 != 0 and self.config['sdxl'] == True:
            self.error_if_unhandled("Bucket step must be a multiple of 32 when using SDXL","Setting bucket step to nearest multiple of 32")
            self.config['bucket_reso_steps'] = to_nearest_multiple(self.config['bucket_reso_steps'],32,raise_to_min=True)
        elif self.config['bucket_reso_steps']%8 != 0:
            self.error_if_unhandled("Bucket step must be a multiple of 8 when using SD 1.5 models or similar","Setting bucket step to nearest multiple of 8")
            self.config['bucket_reso_steps'] = to_nearest_multiple(self.config['bucket_reso_steps'],8,raise_to_min=True)
        if self.config['bucket_reso_steps'] > self.config['base_res']:
            self.error_if_unhandled("Bucket step must be less than or equal to base resolution","Setting bucket step to base resolution")
            self.config['bucket_reso_steps'] = self.config['base_res']
    def validate_all(self):
        self.validate_resolution()
        self.validate_bucket_reso_steps()
        
