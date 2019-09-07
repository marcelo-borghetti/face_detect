import argparse
from keras.models import load_model
# function for face detection with mtcnn
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os


class FaceDetector:
    def __init__(self):
        self.recognizer = load_model('model/facenet_keras.h5')
        # summarize input and output shape
        self.detector = MTCNN()
        self.faces = list()

    def find_face(self, filename, required_size=(160, 160)):
        # load image from file
        image = Image.open(filename)
        image = image.convert('RGB')
        pixels = asarray(image)
        results = self.detector.detect_faces(pixels)
        for i in range(0, len(results)):
            x1, y1, width, height = results[i]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            self.faces.append(image)
        if results:
            return True
        return False

    def save_images(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        for i in range(0, len(self.faces)):
            self.faces[i].save(output_dir + "/" + str(i) + ".jpg", "JPEG")
        return len(self.faces)


def main(filename, output_dir):
    u = FaceDetector()
    u.find_face(filename)
    n_faces = u.save_images(output_dir)
    print('Take a look at', output_dir, 'directory:', n_faces, 'faces found.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", required=True, help="Image input")
    parser.add_argument("-o", "--output_dir", required=False, help="Output directory")
    args = parser.parse_args()
    filename = args.filename
    output_dir = "results"
    if args.output_dir:
        output_dir = args.output_dir
    main(filename, output_dir)
