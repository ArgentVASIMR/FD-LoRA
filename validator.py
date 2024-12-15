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
        if self.config['base_res'] < 512:
            self.error_if_unhandled("Base resolution must be at least 512","Setting base resolution to 512")
            self.config['base_res'] = 512
        if self.config['base_res'] < 1024 and self.config['sdxl'] == True:
            self.error_if_unhandled("Base resolution must be at least 1024 when using SDXL","Setting base resolution to 1024")
            self.config['base_res'] = 1024
    def validate_bucket_step(self):
        if self.config['bucket_step']%32 != 0 and self.config['sdxl'] == True:
            self.error_if_unhandled("Bucket step must be a multiple of 32 when using SDXL","Setting bucket step to nearest multiple of 32")
            self.config['bucket_step'] = self.config['bucket_step'] - (self.config['bucket_step']%32)
        if self.config['bucket_step']%8 != 0 and self.config['sdxl'] == False:
            self.error_if_unhandled("Bucket step must be a multiple of 8 when using SD 1.5 models or similar","Setting bucket step to nearest multiple of 8")
            self.config['bucket_step'] = self.config['bucket_step'] - (self.config['bucket_step']%8)
        if self.config['bucket_step'] > self.config['base_res']:
            self.error_if_unhandled("Bucket step must be less than or equal to base resolution","Setting bucket step to base resolution")
            self.config['bucket_step'] = self.config['base_res']
    def validate_all(self):
        self.validate_resolution()
        self.validate_bucket_step()
        
