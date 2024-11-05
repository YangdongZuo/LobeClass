import cv2
import numpy as np
from PIL import Image
import onnxruntime as rt
import argparse
import json
import os
#import serial.serialposix 


EXPORT_MODEL_VERSION = 1
class ONNXModel:
    def __init__(self, dir_path) -> None:
        """Method to get name of model file. Assumes model is in the parent directory for script."""
        model_dir = os.path.dirname(dir_path)
        with open(os.path.join(model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = os.path.join(model_dir, self.signature.get("filename"))
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file does not exist")
        # get the signature for model inputs and outputs
        self.signature_inputs = self.signature.get("inputs")
        self.signature_outputs = self.signature.get("outputs")
        self.session = None
        if "Image" not in self.signature_inputs:
            raise ValueError("ONNX model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")
        # Look for the version in signature file.
        # If it's not found or the doesn't match expected, print a message
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {EXPORT_MODEL_VERSION}."
            )

    def load(self) -> None:
        """Load the model from path to model file"""
        # Load ONNX model as session.
        self.session = rt.InferenceSession('model.onnx',provider=['TensorrtExecutionProvider'])
        print("sc")
    def predict(self, image: Image.Image) -> dict:
        """
        Predict with the ONNX session!
        """
        print("predict")
        # process image to be compatible with the model
        img = self.process_image(image, self.signature_inputs.get("Image").get("shape"))
        # run the model!
        fetches = [(key, value.get("name")) for key, value in self.signature_outputs.items()]
        # make the image a batch of 1
        feed = {self.signature_inputs.get("Image").get("name"): [img]}
        outputs = self.session.run(output_names=[name for (_, name) in fetches], input_feed=feed)
        return self.process_output(fetches, outputs)

    def process_image(self, image: Image.Image, input_shape: list) -> np.ndarray:
        """
        Given a PIL Image, center square crop and resize to fit the expected model input, and convert from [0,255] to [0,1] values.
        """
        width, height = image.size
        print("process image")
        # ensure image type is compatible with model and convert if not
        if image.mode != "RGB":
            image = image.convert("RGB")
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        # make 0-1 float instead of 0-255 int (that PIL Image loads by default)
        image = np.asarray(image) / 255.0
        # format input as model expects
        return image.astype(np.float32)

    def process_output(self, fetches: dict, outputs: dict) -> dict:
        # un-batch since we ran an image with batch size of 1,
        # convert to normal python types with tolist(), and convert any byte strings to normal strings with .decode()
        out_keys = ["label", "confidence"]
        results = {}
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output
    

    def infer(self, frame)->dict:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 进行推理
        outputs = model.predict(pil_image)
        # 获取置信度最高的标签
        predicted_labels = outputs["predictions"]
        if predicted_labels:
            top_prediction = predicted_labels[0]
            label = top_prediction["label"]
            confidence = top_prediction["confidence"]
            print(f"label:{label},confidence:{confidence}")
        return label


if __name__ == "__main__":
    # receve_ser=serial.serialposix.Serial('/dev/ttyUSB1',9600,timeout=1)
    # send_ser = serial.serialposix.Serial('/dev/ttyUSB0',9600,timeout=1)
    print("信号OK")
    # signal=receve_ser.read().decode('ASCII')
    signal_history='no signal'

    parser = argparse.ArgumentParser(description="Capture and classify images from the camera.")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index to use (default: 0).")
    args = parser.parse_args()
    #dir_path = os.getcwd()  ##获取当前目录
    dir_path = "model.onnx"  ##获取指定
    # 初始化摄像头
    camera = cv2.VideoCapture(args.camera_index)
    print("摄像头OK")
    # 初始化 Lobe 导出的模型
    model = ONNXModel(dir_path=r"C:\Users\32960\Desktop\test\CLASS_TEST\model.onnx")
    model.load()
    if True:
        # signal=receve_ser.read().decode('ASCII')
        print("receving signal")
        # # 避免反复操作
        # if signal_history == signal:
        #     # continue
        # else:
        #     signal_history = signal
        # if signal=='D':
            # 从摄像头捕获图像
        ret, frame = camera.read()
        if not ret:
            print("无法读取摄像头帧")
            exit()
        #frame=cv2.imread(r'C:\Users\32960\Desktop\test\CLASS_TEST\0.jpg')
        model.infer(frame)
        cv2.namedWindow("1",cv2.WINDOW_FREERATIO)
        cv2.imshow("1", frame)

            # if label=="cans" or label=="bottles":
                #     send_ser.write('0'.encode())#test_car farword
                #     response = send_ser.readall()#read a string from port
                #     print(response)
                # elif label=="carrot" or "potato" or "turnip":
                #     send_ser.write('1'.encode())#test_car farword
                #     response = send_ser.readall()#read a string from port
                #     print(response)
                # elif label=="battery" or "medecine" :
                #     send_ser.write('2'.encode())#test_car farword
                #     response = send_ser.readall()#read a string from port
                #     print(response)
                # elif label=="china" or "cobble":
                #     send_ser.write('3'.encode())#test_car farword
                #     response = send_ser.readall()#read a string from port
                #     print(response)
                # else:
                #     send_ser.write('F'.encode())#test_car stop
        # 在窗口中显示摄像头捕获的图像

                # break
        # signal=receve_ser.read().decode('ASCII')
        # if signal == 'K':
        #     print("可回收已满")
        # elif signal == 'Q':
        #     print("其他垃圾已满")
        # elif signal == 'C':
        #     print("厨余垃圾已满")
        # elif signal == 'Y':
        #     print("有害垃圾已满")
            # 按 'q' 键退出循环
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
        # 关闭摄像头和窗口
    camera.release()
    cv2.destroyAllWindows()

