# resnet_50_schoolbus

## Steps

1. Activate virtual environment and install dependencies
2. Download [Google protocol buffers](https://github.com/protocolbuffers/protobuf/releases). I installed `protoc-25.1-win32.zip`
3. Edit `model_downloader.py` for the desired model, then run to download an extract
4. Run script with desired model.  Example: `python .\detect_from_webcam.py -m ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model -l mscoco_label_map.pbtxt`