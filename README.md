# Llama.cpp Model Controller 🦙

A simple, modern web interface for managing and deploying llama.cpp language models with ease.

![Llama.cpp Web UI Screenshot](https://api.placeholder.com/400/300)

## 🌟 Features

- **Simple Model Management** - Start and stop models with a clean, intuitive interface
- **Real-time GPU Monitoring** - Track GPU temperature, memory usage and utilization
- **Live Server Logs** - View token usage and server output in real-time with color-coded logs
- **Customizable Parameters** - Configure threads, context size, GPU layers and more
- **Multi-GPU Support** - Choose between layer and row split modes for optimal multi-GPU performance
- **Responsive Design** - Works on desktop and mobile devices
- **Per-Model Configuration** - Save and load different settings for each model

## 📋 Requirements

- Linux environment with CUDA support
- NVIDIA GPU(s) with appropriate drivers
- Python 3.8+ 
- llama.cpp compiled with CUDA support
- GGUF format models

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dan-Duran/llama-cpp-model-controller.git
cd llama-cpp-model-controller
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install flask
```

### 4. Set up model directory

By default, the application looks for models in a `models` directory. Make sure your GGUF models are placed there:

```bash
mkdir -p models
# Copy or download your GGUF models to the models directory
```

### 5. Configure the application

Edit `app.py` to customize the path variables at the top of the file:

```python
# Configuration variables - update these to match your environment
HOME_DIR = os.path.expanduser("~")  # Dynamically get user's home directory
PROJECT_DIR = os.path.join(HOME_DIR, "llama")  # Project directory
MODEL_DIR = os.path.join(PROJECT_DIR, "models")  # Model directory
LLAMA_CPP_PATH = os.path.join(PROJECT_DIR, "llama.cpp-b4823/build/bin/llama-server")  # Path to llama-server executable
CACHE_DIR = os.path.join(HOME_DIR, ".cache/llama")  # Llama cache directory
```

You can modify these paths to match your specific environment:
- `HOME_DIR`: Your home directory (automatically detected)
- `PROJECT_DIR`: The directory where the application is installed
- `MODEL_DIR`: Where your GGUF models are stored
- `LLAMA_CPP_PATH`: Path to the llama-server executable
- `CACHE_DIR`: Where llama.cpp stores its cache files


## 🖥️ Usage

### Starting the Web UI

```bash
python app.py
```

The web interface will be available at `http://localhost:5000` by default.

### Deploying a Model

1. Select a model from the dropdown menu
2. Configure the parameters (or use the defaults)
   - **Threads**: Number of CPU threads to use (recommended: 16)
   - **GPU Layers**: Number of layers to offload to GPU (default: 99)
   - **Context Size**: Token context window size (default: 32000)
   - **Split Mode**: How to divide the model across GPUs (layer or row)
   - **Parallel Sequences**: Number of sequences to process in parallel
3. Click "Start Model"
4. The model will be available at the configured host:port

### Monitoring

- The GPU stats section shows current GPU utilization
- The Server Logs section displays real-time output from the llama-server process
- Color-coded logs help identify errors, warnings, and token usage information

### Stopping a Model

Click the "Stop Model" button to terminate the server and clear the cache.

## 📁 Project Structure

```
llama-cpp-web-ui/
├── app.py                  # Flask application
├── templates/              # HTML templates
│   └── index.html          # Main UI template
├── models/                 # GGUF model files (not included in repo)
└── venv/                   # Python virtual environment
```

## ⚙️ Advanced Configuration

### Custom Port

To change the port the web UI runs on, modify the last line in `app.py`:

```python
app.run(host="0.0.0.0", port=5000, debug=True)
```

### Adding Model Presets

You can create preset configurations for each model by modifying the HTML template.

## 🔧 Troubleshooting

### Common Issues

1. **Model fails to load**
   - Ensure you have enough GPU memory
   - Try reducing context size or using fewer GPU layers
   - Check model file permissions
   - Verify the `MODEL_DIR` path is correct and models exist at that location

2. **GPU not detected**
   - Verify CUDA installation
   - Check nvidia-smi output
   - Ensure llama.cpp was compiled with CUDA support

3. **Out of memory errors**
   - Reduce context size
   - Use a smaller model
   - Adjust split mode settings

4. **Llama-server executable not found**
   - Check that the `LLAMA_CPP_PATH` points to your compiled llama-server executable
   - Make sure llama.cpp is compiled with CUDA support
   - Verify execution permissions with `chmod +x [path-to-llama-server]`

5. **Path configuration issues**
   - The application logs the paths it's using on startup - check these logs
   - Ensure all directories exist and are accessible
   - If running in a container or different environment, update paths accordingly

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🙏 Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the incredible optimized LLM implementation
- All the model creators and fine-tuners who make their work available