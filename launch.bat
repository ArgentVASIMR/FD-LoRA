call D:\StableDiffusionProjects\sd-scripts\venv\scripts\activate.bat

set script_dir=D:\StableDiffusionProjects\sd-scripts
set yaml_list=base.yaml hyperparameters.yaml directories.yaml extras.yaml performance.yaml advanced_parameters.yaml

accelerate launch --num_cpu_threads_per_process 8 main.py --script_dir %script_dir% --yaml_list %yaml_list%
pause