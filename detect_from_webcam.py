import numpy as np
import argparse
import tensorflow as tf
import cv2

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]
    
    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    # if 'detection_masks' in output_dict:
    #     # Reframe the the bbox mask to the image size.
    #     detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
    #                                 output_dict['detection_masks'], output_dict['detection_boxes'],
    #                                 image.shape[0], image.shape[1])      
    #     detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
    #     output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    # return output_dict

def parse_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, 'r') as file:
        for line in file:
            if 'id:' in line:
                id = int(line.strip().split(' ')[-1])
            if 'display_name:' in line:
                name = line.strip().split(' ')[-1].strip('"')
                label_map[id] = name
    return label_map

def visualize_output(image_np, boxes, classes, scores, category_index, instance_masks=None, use_normalized_coordinates=True, line_thickness=8):
    for i in range(len(scores)):
        if scores[i] > 0.5:  # You can adjust the threshold
            box = boxes[i]
            if use_normalized_coordinates:
                box = tuple(box.tolist())
                im_height, im_width, _ = image_np.shape
                (left, right, top, bottom) = (box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)
            else:
                (left, right, top, bottom) = (box[1], box[3], box[0], box[2])

            cv2.rectangle(image_np, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), line_thickness)
            class_name = category_index[classes[i]]
            label = f"{class_name}: {int(scores[i] * 100)}%"
            cv2.putText(image_np, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add mask visualization here if necessary
            # if instance_masks is not None:
            #     mask = instance_masks[i]
            #     image_np = apply_mask(image_np, mask)

    return image_np

def run_inference(model, category_index, cap):
    while True:
        ret, image_np = cap.read()
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        visualize_output(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8
        )
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    args = parser.parse_args()

    detection_model = load_model(args.model)
    category_index = parse_label_map(args.labelmap)

    # Internal Webcam
    cap = cv2.VideoCapture(0)

    ### Hikvision IPCAM
    #cap = cv2.VideoCapture()
    #cap.open("rtsp://admin:password@192.168.1.199:10554/Streaming/Channels/301")

    run_inference(detection_model, category_index, cap)


# Faster Model
# python .\detect_from_webcam.py -m ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model -l mscoco_label_map.pbtxt

# Slower Model
# python .\detect_from_webcam.py -m faster_rcnn_resnet50_v1_640x640_coco17_tpu-8\saved_model -l mscoco_label_map.pbtxt
