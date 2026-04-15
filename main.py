import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Train an image classification model.")
parser.add_argument(
    "--framework",
    choices=["pytorch", "tensorflow"],
    required=True,
    help="Framework to use for training: 'pytorch' or 'tensorflow'"
)
args = parser.parse_args()

if args.framework == "pytorch":
    print("Starting PyTorch training...")
    subprocess.run([sys.executable, "pytorch_model.py"], check=True)
elif args.framework == "tensorflow":
    print("Starting TensorFlow training...")
    subprocess.run([sys.executable, "tensorflow_model.py"], check=True)
