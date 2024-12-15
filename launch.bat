call D:\StableDiffusionProjects\sd-scripts\venv\scripts\activate.bat

set script_dir=D:\StableDiffusionProjects\sd-scripts
set yaml_list=config.yaml hyperparameters.yaml directories.yaml options.yaml performance.yaml

accelerate launch --num_cpu_threads_per_process 8 main.py --script_dir %script_dir% --yaml_list %yaml_list%
pause