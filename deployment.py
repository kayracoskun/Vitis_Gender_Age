import sys
import numpy as np
import cv2
import vitis_ai_library
import xir
import time
import os

# Check if the model name argument is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <model_name>")
    sys.exit(1)

DPU_CONFIG = sys.argv[1]

# file path
MODEL_PATH = f"./xmodel_outputs/{DPU_CONFIG}/{DPU_CONFIG}.xmodel"
IMAGES_FOLDER = "./images/"

all_process_start_time = time.time()

g = xir.Graph.deserialize(MODEL_PATH)
runner = vitis_ai_library.GraphRunner.create_graph_runner(g)

# input buffer
inputDim = tuple(runner.get_inputs()[0].get_tensor().dims)
inputData = [np.empty(inputDim, dtype=np.int8)]

image_filenames = [f for f in os.listdir(IMAGES_FOLDER) if os.path.isfile(os.path.join(IMAGES_FOLDER, f))]
num_images = len(image_filenames)  # Get the number of images dynamically

total_inference_duration_ms = 0

for i, filename in enumerate(image_filenames):
    IMG_PATH = os.path.join(IMAGES_FOLDER, filename)

    # input image
    #image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(IMG_PATH)

    # normalization
    image = image / 255.0

    # quantization
    fix_point = runner.get_input_tensors()[0].get_attr("fix_point")
    scale = 2 ** fix_point
    image = (image * scale).round()
    image = image.astype(np.int8)

    # set input data
    #inputData[0][0] = image.reshape(28, 28, 1)
    inputData[0][0] = image

    # Start timing for inference
    inference_start_time = time.time()

    # output buffer
    outputData = runner.get_outputs()

    # prediction
    job_id = runner.execute_async(inputData, outputData)
    runner.wait(job_id)

    inference_end_time = time.time()

    # Inference duration for this image
    inference_duration_ms = (inference_end_time - inference_start_time) * 1000
    total_inference_duration_ms += inference_duration_ms

    resultList = np.asarray(outputData[0])[0]
    resultIdx = resultList.argmax()
    resultVal = resultList[resultIdx]

    # Print only the most accurate value
    print(f"Most accurate prediction for image {i}: {resultIdx} with value {resultVal:.5f}")

average_inference_time_ms = total_inference_duration_ms / num_images

print(f"Average inference duration: {average_inference_time_ms:.5f} ms")
print(f"Total inference duration: {total_inference_duration_ms:.5f} ms")

all_process_end_time = time.time()
all_process_duration_ms = (all_process_end_time - all_process_start_time) * 1000
print(f"All process duration: {all_process_duration_ms:.5f} ms")

del runner