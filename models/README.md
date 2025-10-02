## Model Download & Setup Instructions

1. **Download the Model Files**
   - Visit the Hugging Face repository: [https://huggingface.co/alvanalrakib/Aurora-Labs](https://huggingface.co/alvanalrakib/Aurora-Labs)
   - Download the model file (e.g., `melody_model.safetensors`) **and** its corresponding JSON config file (e.g., `melody_model_config.json`).

2. **Model File Placement**
   - Ensure that **both** the model file (e.g., `melody_model.safetensors`) and its corresponding JSON config file (e.g., `melody_model_config.json`) are placed **together in the same directory**. 
   - You may use the default `models/` directory provided in this repository, or specify a custom directory of your choice. The backend will automatically scan the specified directory for valid model/config pairs.
   - **Tip:** Keeping your models organized in subfolders (e.g., `models/melody/`, `models/drums/`) is supported and can help manage multiple models.

   - **Configuration:**  
     - By default, the backend will auto-discover models from the directory specified in `config/models.yaml` under `storage.models_root`.
     - If you wish to use a custom location, update the `models_root` path in your `config/models.yaml` file accordingly.  
     - Example:
       ```yaml
       storage:
         models_root: "./models"  # Change this path if your models are elsewhere
       ```

3. **Install the Tokenizer**
   - Download the tokenizer wheel file (`alv_tokenizer-2.0.1-py3-none-any.whl`) from the same Hugging Face repository.
   - If you are using a virtual environment (venv) or conda environment, **activate it first** to ensure the tokenizer is installed in the correct environment.
   - Install the tokenizer by running:
     ```
     pip install path/to/alv_tokenizer-2.0.1-py3-none-any.whl
     ```
   - Make sure the `.whl` is installed in the same environment where you will run the Aurora Backend.

**Note:**  
- Always keep the model and its config JSON together in the same directory.
- If you update or move the model, ensure the JSON config moves with it.