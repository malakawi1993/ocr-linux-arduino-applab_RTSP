#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

if [ ! -d .venv ]; then
    echo "Installing Python dependencies..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    echo "Installing Python dependencies OK"
    echo ""
else
    source .venv/bin/activate
fi

if [ ! -f det_v5.onnx ]; then
    echo "Downloading det_v5.onnx..."
    wget -O det_v5.onnx https://huggingface.co/monkt/paddleocr-onnx/resolve/436f75fa5a51ecc6b1d27892684d9c49ac8600c0/detection/v5/det.onnx
    echo "Downloading det_v5.onnx OK"
    echo ""
fi

if [ ! -f det_v3.onnx ]; then
    echo "Downloading det_v3.onnx..."
    wget -O det_v3.onnx https://huggingface.co/monkt/paddleocr-onnx/resolve/436f75fa5a51ecc6b1d27892684d9c49ac8600c0/detection/v3/det.onnx
    echo "Downloading det_v3.onnx OK"
    echo ""
fi

if [ ! -f rec.onnx ]; then
    echo "Downloading rec.onnx..."
    wget -O rec.onnx https://huggingface.co/monkt/paddleocr-onnx/resolve/436f75fa5a51ecc6b1d27892684d9c49ac8600c0/languages/english/rec.onnx
    echo "Downloading rec.onnx OK"
    echo ""
fi

if [ ! -f rec_en_dict.txt ]; then
    echo "Downloading rec_en_dict.txt..."
    wget -O rec_en_dict.txt https://huggingface.co/monkt/paddleocr-onnx/resolve/436f75fa5a51ecc6b1d27892684d9c49ac8600c0/languages/english/dict.txt
    echo "Downloading rec_en_dict.txt OK"
    echo ""
fi

if [ ! -d openimages/car ]; then
    echo "Downloading openimages..."
    oi_download_images --base_dir=openimages --labels Car --limit 200
    echo "Downloading openimages OK"
    echo ""
fi

echo "Creating representative datasets..."
python3 create_representative_dataset.py --width 640 --height 480 --limit 30
python3 create_representative_dataset.py --width 320 --height 48 --limit 100
echo "Creating representative datasets OK"
echo ""

echo "All done! Now upload det_v5_480_640.onnx / det_v3_480_640.onnx / rec_48_320.onnx through BYOM."
