import os
import subprocess
import signal
import atexit
from flask import Flask, render_template, request, jsonify

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
        model = request.form.get("model")
        print(f"DEBUG: Received model - {model}")  # 👈 Add this to see what is received

        if not model:
            return jsonify({"status": "No model selected!", "success": False})

        threads = request.form.get("threads", "16")
        port = request.form.get("port", "8080")
        host = request.form.get("host", "0.0.0.0")
        ngl = request.form.get("ngl", "99")
        c = request.form.get("c", "32000")
        sm = request.form.get("sm", "row")
        np = request.form.get("np", "1")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0,1"

        command = f"""
        CUDA_VISIBLE_DEVICES=0,1 /home/dan/llama/llama.cpp-b4823/build/bin/llama-server \
        -m {MODEL_DIR}/{model} --threads {threads} --port {port} --host {host} \
        -ngl {ngl} -c {c} -sm {sm} -np {np}
        """

        model_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid, env=env)

        return jsonify({"status": f"Model '{model}' started on {host}:{port}", "success": True})

    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}", "success": False})

@app.route("/stop", methods=["POST"])
def stop_server():
    global model_process

    if model_process:
        os.killpg(os.getpgid(model_process.pid), signal.SIGTERM)
        model_process = None

    # 🛑 Forcefully Clear Llama-Server Cache (Modify This for Your Setup)
    try:
        subprocess.run(["rm", "-rf", "/home/dan/.cache/llama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Llama cache cleared.")
    except Exception as e:
        return jsonify({"status": f"Error clearing cache: {str(e)}", "success": False})

    # Double-check: Kill any remaining `llama-server` processes
    subprocess.run(["pkill", "-f", "llama-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return jsonify({"status": "Model server fully stopped and cache cleared!", "success": True})



@app.route("/status")
def status():
    return jsonify({"running": model_process is not None})

# Ensure the model process is killed when Flask exits
def cleanup():
    global model_process
    print("Cleaning up before exit...")

    if model_process:
        os.killpg(os.getpgid(model_process.pid), signal.SIGTERM)
        print("Model process terminated.")

    # 🛑 Clear Llama-Server Cache on Exit
    subprocess.run(["rm", "-rf", "/home/dan/.cache/llama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Llama cache cleared.")

    # Kill any remaining `llama-server` processes
    subprocess.run(["pkill", "-f", "llama-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


atexit.register(cleanup)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
