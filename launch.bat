call D:\StableDiffusionProjects\sd-scripts\venv\scripts\activate.bat
accelerate launch --num_cpu_threads_per_process 8 run_antrogent.py
pause