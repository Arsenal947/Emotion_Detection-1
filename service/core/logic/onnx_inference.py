import cv2
import numpy as np
import time 
import tflite_runtime.interpreter as tflite

def emotions_detecter(image_array):
    if len(image_array.shape)== 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    t1 = time.time()
    test_image = cv2.resize(image_array, (256, 256))

    image_array = np.expand_dims(test_image, axis = 0)


    interpreter = tflite.Interpreter(model_path = 'service/eff_model.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details['index'], image_array)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details['index'])[0]
    time_elapsed = time.time()-t1
    emotion = ''
    if np.argmax(output) == 0:
        emotion = 'angry'
    elif np.argmax(output) == 1:
        emotion = 'sad'
    else:
        emotion = 'happy'
    return {
        "emotion": emotion,
        "time_elapsed": str(time_elapsed)
    }