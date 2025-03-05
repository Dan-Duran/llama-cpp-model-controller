import os
import subprocess
import signal
import atexit
import logging
from flask import Flask, render_template, request, jsonify

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
model_process = None
MODEL_DIR = "/home/dan/llama/models"

def get_models():
    """Retrieve all available models from the models directory"""
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]

def get_gpu_stats():
    """Run nvidia-smi and extract key GPU usage stats."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], 
                                stdout=subprocess.PIPE, text=True)
        gpus = []
        for line in result.stdout.strip().split("\n"):
            index, name, temp, usage, mem_used, mem_total = line.split(", ")
            gpus.append({
                "index": index,
                "name": name,
                "temp": f"{temp}°C",
                "usage": f"{usage}%",
                "memory": f"{mem_used}MB / {mem_total}MB"
            })
        return gpus
    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
        return [{"error": str(e)}]

@app.route("/")
def index():
    models = get_models()
    return render_template("index.html", models=models)

@app.route("/gpu")
def gpu_stats():
    return jsonify(get_gpu_stats())

@app.route("/start", methods=["POST"])
def start_server():
    global model_process
    if model_process:
        return jsonify({"status": "Model is already running!", "success": False})

    try:
        # Debug all form data
        logger.debug("Form data received:")
        for key, value in request.form.items():
            logger.debug(f"  {key}: {value}")
        
        # Get model name
        model = request.form.get("model")
        if not model:
            logger.error("No model selected!")
            return jsonify({"status": "No model selected!", "success": False})
        
        # Get other parameters with defaults
        threads = request.form.get("threads", "16")
        port = request.form.get("port", "8080")
        host = request.form.get("host", "0.0.0.0")
        ngl = request.form.get("ngl", "99")
        c = request.form.get("c", "32000")
        sm = request.form.get("sm", "layer")
        np = request.form.get("np", "1")
        
        # Log the actual values being used
        logger.debug(f"Using model: {model}")
        logger.debug(f"Threads: {threads}")
        logger.debug(f"Port: {port}")
        logger.debug(f"Host: {host}")
        logger.debug(f"GPU Layers (ngl): {ngl}")
        logger.debug(f"Context size (c): {c}")
        logger.debug(f"Split mode (sm): {sm}")
        logger.debug(f"Parallel sequences (np): {np}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0,1"

        command = f"""
        CUDA_VISIBLE_DEVICES=0,1 /home/dan/llama/llama.cpp-b4823/build/bin/llama-server \
        -m {MODEL_DIR}/{model} --threads {threads} --port {port} --host {host} \
        -ngl {ngl} -c {c} -sm {sm} -np {np}
        """
        
        # Log the final command being executed
        logger.debug(f"Executing command: {command}")

        model_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid, env=env)
        
        # Capture any immediate output from the process
        stdout_data, stderr_data = model_process.communicate(timeout=0.5)
        if stdout_data:
            logger.debug(f"Process stdout: {stdout_data.decode('utf-8')}")
        if stderr_data:
            logger.debug(f"Process stderr: {stderr_data.decode('utf-8')}")
        
        return jsonify({"status": f"Model '{model}' started on {host}:{port}", "success": True})

    except subprocess.TimeoutExpired:
        # This is actually expected - the process should keep running
        logger.debug("Process started successfully (timeout expected)")
        return jsonify({"status": f"Model '{model}' started on {host}:{port}", "success": True})
    except Exception as e:
        logger.exception(f"Error starting model: {e}")
        return jsonify({"status": f"Error: {str(e)}", "success": False})

@app.route("/stop", methods=["POST"])
def stop_server():
    global model_process

    logger.debug("Stopping model server...")
    if model_process:
        os.killpg(os.getpgid(model_process.pid), signal.SIGTERM)
        model_process = None
        logger.debug("Model process terminated")

    # Clear Llama-Server Cache
    try:
        subprocess.run(["rm", "-rf", "/home/dan/.cache/llama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.debug("Llama cache cleared")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"status": f"Error clearing cache: {str(e)}", "success": False})

    # Double-check: Kill any remaining `llama-server` processes
    subprocess.run(["pkill", "-f", "llama-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.debug("Killed any remaining llama-server processes")

    return jsonify({"status": "Model server fully stopped and cache cleared!", "success": True})

@app.route("/status")
def status():
    return jsonify({"running": model_process is not None})

# Ensure the model process is killed when Flask exits
def cleanup():
    global model_process
    logger.info("Cleaning up before exit...")

    if model_process:
        os.killpg(os.getpgid(model_process.pid), signal.SIGTERM)
        logger.info("Model process terminated")

    # Clear Llama-Server Cache on Exit
    subprocess.run(["rm", "-rf", "/home/dan/.cache/llama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info("Llama cache cleared")

    # Kill any remaining `llama-server` processes
    subprocess.run(["pkill", "-f", "llama-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info("Killed any remaining llama-server processes")

atexit.register(cleanup)

if __name__ == "__main__":
    logger.info("Starting Llama Model Controller server...")
    app.run(host="0.0.0.0", port=5000, debug=True)