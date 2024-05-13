IS_DLIB_INSTALLED = False
try:
    import dlib
    IS_DLIB_INSTALLED = True
except ImportError:
    pass

IS_INSIGHTFACE_INSTALLED = False
try:
    from insightface.app import FaceAnalysis
    IS_INSIGHTFACE_INSTALLED = True
except ImportError:
    pass

if not IS_DLIB_INSTALLED and not IS_INSIGHTFACE_INSTALLED:
    raise Exception("Please install either dlib or insightface to use this node.")

import torch
#import torch.nn.functional as F
import torchvision.transforms.v2 as T
#import comfy.utils
import os
import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

DLIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlib")
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

THRESHOLDS = { # from DeepFace
        "VGG-Face": {"cosine": 0.68, "euclidean": 1.17, "L2_norm": 1.17},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "L2_norm": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "L2_norm": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "L2_norm": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "L2_norm": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "L2_norm": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "L2_norm": 0.55},
        "DeepFace": {"cosine": 0.23, "euclidean": 64, "L2_norm": 0.64},
        "DeepID": {"cosine": 0.015, "euclidean": 45, "L2_norm": 0.17},
        "GhostFaceNet": {"cosine": 0.65, "euclidean": 35.71, "L2_norm": 1.10},
    }

def tensor_to_image(image):
    return np.array(T.ToPILImage()(image.permute(2, 0, 1)).convert('RGB'))

def image_to_tensor(image):
    return T.ToTensor()(image).permute(1, 2, 0)
    #return T.ToTensor()(Image.fromarray(image)).permute(1, 2, 0)

class InsightFace:
    def __init__(self, provider="CPU", name="buffalo_l"):
        self.face_analysis = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.thresholds = THRESHOLDS["ArcFace"]

    def get_face(self, image):
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.face_analysis.det_model.input_size = size
            faces = self.face_analysis.get(image)
            if len(faces) > 0:
                return sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
        return None

    def get_embeds(self, image):
        face = self.get_face(image)
        if face is not None:
            face = face[0].normed_embedding
        return face
    
    def get_bbox(self, image, padding=0, padding_percent=0):
        faces = self.get_face(np.array(image))
        img = []
        x = []
        y = []
        w = []
        h = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            width = x2 - x1
            height = y2 - y1
            x1 = int(max(0, x1 - int(width * padding_percent) - padding))
            y1 = int(max(0, y1 - int(height * padding_percent) - padding))
            x2 = int(min(image.width, x2 + int(width * padding_percent) + padding))
            y2 = int(min(image.height, y2 + int(height * padding_percent) + padding))
            crop = image.crop((x1, y1, x2, y2))
            img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
            x.append(x1)
            y.append(y1)
            w.append(x2 - x1)
            h.append(y2 - y1)
        return (img, x, y, w, h)

class DLib:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        # check if the models are available
        if not os.path.exists(os.path.join(DLIB_DIR, "shape_predictor_5_face_landmarks.dat")):
            raise Exception("The 5 point landmark model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/shape_predictor_5_face_landmarks.dat")
        if not os.path.exists(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat")):
            raise Exception("The face recognition model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/dlib_face_recognition_resnet_model_v1.dat")

        self.shape_predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_5_face_landmarks.dat"))
        self.face_recognition = dlib.face_recognition_model_v1(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat"))
        self.thresholds = THRESHOLDS["Dlib"]

    def get_face(self, image):
        faces = self.face_detector(np.array(image), 1)
        if len(faces) > 0:
            return sorted(faces, key=lambda x: x.area(), reverse=True)
        return None

    def get_embeds(self, image):
        faces = self.get_face(image)
        if faces is not None:
            shape = self.shape_predictor(image, faces[0])
            faces = np.array(self.face_recognition.compute_face_descriptor(image, shape))
        return faces
    
    def get_bbox(self, image, padding=0, padding_percent=0):
        faces = self.get_face(image)
        img = []
        x = []
        y = []
        w = []
        h = []
        for face in faces:
            x1 = max(0, face.left() - int(face.width() * padding_percent) - padding)
            y1 = max(0, face.top() - int(face.height() * padding_percent) - padding)
            x2 = min(image.width, face.right() + int(face.width() * padding_percent) + padding)
            y2 = min(image.height, face.bottom() + int(face.height() * padding_percent) + padding)
            crop = image.crop((x1, y1, x2, y2))
            img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
            x.append(x1)
            y.append(y1)
            w.append(x2 - x1)
            h.append(y2 - y1)
        return (img, x, y, w, h)
    
    def get_landmarks(self, image):
        faces = self.get_face(image)
        if faces is not None:
            shape = self.shape_predictor(image, faces[0])
            return shape
        return None

class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(s):
        libraries = []
        if IS_INSIGHTFACE_INSTALLED:
            libraries.append("insightface")
        if IS_DLIB_INSTALLED:
            libraries.append("dlib")

        return {"required": {
            "library": (libraries, ),
            "provider": (["CPU", "CUDA", "DirectML", "OpenVINO", "ROCM", "CoreML"], ),
        }}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "FaceAnalysis"

    def load_models(self, library, provider):
        out = {}

        if library == "insightface":
            out = InsightFace(provider)
        else:
            out = DLib()

        return (out, )

class FaceBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "padding_percent": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05 }),
                "index": ("INT", { "default": -1, "min": -1, "max": 4096, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "x", "y", "width", "height")
    FUNCTION = "bbox"
    CATEGORY = "FaceAnalysis"
    OUTPUT_IS_LIST = (True, True, True, True, True,)

    def bbox(self, analysis_models, image, padding, padding_percent, index=-1):
        out_img = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []

        for i in image:
            i = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')
            img, x, y, w, h = analysis_models.get_bbox(i, padding, padding_percent)
            out_img.extend(img)
            out_x.extend(x)
            out_y.extend(y)
            out_w.extend(w)
            out_h.extend(h)

        if not out_img:
            raise Exception('No face detected in image.')

        if len(out_img) == 1:
            index = 0

        if index > len(out_img) - 1:
            index = len(out_img) - 1

        if index != -1:
            out_img = [out_img[index]]
            out_x = [out_x[index]]
            out_y = [out_y[index]]
            out_w = [out_w[index]]
            out_h = [out_h[index]]
        #else:
        #    w = out_img[0].shape[1]
        #    h = out_img[0].shape[0]

            #out_img = [comfy.utils.common_upscale(img.unsqueeze(0).movedim(-1,1), w, h, "bilinear", "center").movedim(1,-1).squeeze(0) for img in out_img]
            #out_img = torch.stack(out_img)
        
        return (out_img, out_x, out_y, out_w, out_h,)

class FaceEmbedDistance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "reference": ("IMAGE", ),
                "image": ("IMAGE", ),
                "similarity_metric": (["L2_norm", "cosine", "euclidean"], ),
                "filter_thresh": ("FLOAT", { "default": 100.0, "min": 0.001, "max": 100.0, "step": 0.001 }),
                "filter_best": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "generate_image_overlay": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "distance")
    FUNCTION = "analize"
    CATEGORY = "FaceAnalysis"

    def analize(self, analysis_models, reference, image, similarity_metric, filter_thresh, filter_best, generate_image_overlay=True):
        if generate_image_overlay:
            font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"), 32)
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        if filter_thresh == 0.0:
            filter_thresh = analysis_models.thresholds[similarity_metric]

        # you can send multiple reference images in which case the embeddings are averaged
        ref = []
        for i in reference:
            ref_emb = analysis_models.get_embeds(np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')))
            if ref_emb is not None:
                ref.append(torch.from_numpy(ref_emb))
        
        if ref == []:
            raise Exception('No face detected in reference image')

        ref = torch.stack(ref)
        ref = np.array(torch.mean(ref, dim=0))

        out = []
        out_dist = []
        
        for i in image:
            img = np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB'))

            img = analysis_models.get_embeds(img)

            if img is None: # No face detected
                dist = 100.0
                norm_dist = 0
            else:
                if np.array_equal(ref, img): # Same face
                    dist = 0.0
                    norm_dist = 1
                else:
                    if similarity_metric == "L2_norm":
                        #dist = euclidean_distance(ref, img, True)
                        ref = ref / np.linalg.norm(ref)
                        img = img / np.linalg.norm(img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    elif similarity_metric == "cosine":
                        dist = np.float64(1 - np.dot(ref, img) / (np.linalg.norm(ref) * np.linalg.norm(img)))
                        #dist = cos_distance(ref, img)
                    else:
                        #dist = euclidean_distance(ref, img)
                        dist = np.float64(np.linalg.norm(ref - img))
                    
                    norm_dist = min(1.0, 1 / analysis_models.thresholds[similarity_metric] * dist)
           
            if dist <= filter_thresh:
                print(f"\033[96mFace Analysis: value: {dist}, normalized: {norm_dist}\033[0m")

                if generate_image_overlay:
                    tmp = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGBA')
                    txt = Image.new('RGBA', (image.shape[2], txt_height), color=background_color)
                    draw = ImageDraw.Draw(txt)
                    draw.text((0, 0), f"VALUE: {round(dist, 3)} | DIST: {round(norm_dist, 3)}", font=font, fill=(255, 255, 255, 255))
                    composite = Image.new('RGBA', tmp.size)
                    composite.paste(txt, (0, tmp.height - txt.height))
                    composite = Image.alpha_composite(tmp, composite)
                    out.append(T.ToTensor()(composite).permute(1, 2, 0))
                else:
                    out.append(i)

                out_dist.append(dist)

        if not out:
            raise Exception('No image matches the filter criteria.')

        # filter out the best matches
        if filter_best > 0:
            out = np.array(out)
            out_dist = np.array(out_dist)
            idx = np.argsort(out_dist)
            out = torch.from_numpy(out[idx][:filter_best])
            out_dist = out_dist[idx][:filter_best].tolist()

        if isinstance(out, list):
            out = torch.stack(out)

        return(out, out_dist,)

class FaceAlign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image_from": ("IMAGE", ),
            }, "optional": {
                "image_to": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "align"
    CATEGORY = "FaceAnalysis"

    def align(self, analysis_models, image_from, image_to=None):
        image_from = tensor_to_image(image_from[0])
        shape = analysis_models.get_landmarks(image_from)
        r_eye_from = (
            int((shape.part(2).x + shape.part(3).x) // 2),
            int((shape.part(2).y + shape.part(3).y) // 2)
        )
        l_eye_from = (
            int((shape.part(0).x + shape.part(1).x) // 2),
            int((shape.part(0).y + shape.part(1).y) // 2)
        )
        angle = float(np.degrees(np.arctan2(l_eye_from[1] - r_eye_from[1], l_eye_from[0] - r_eye_from[0])))

        if image_to is not None:
            image_to = tensor_to_image(image_to[0])
            shape = analysis_models.get_landmarks(image_to)
            r_eye_to = (
                int((shape.part(2).x + shape.part(3).x) // 2),
                int((shape.part(2).y + shape.part(3).y) // 2)
            )
            l_eye_to = (
                int((shape.part(0).x + shape.part(1).x) // 2),
                int((shape.part(0).y + shape.part(1).y) // 2)
            )
            angle -= float(np.degrees(np.arctan2(l_eye_to[1] - r_eye_to[1], l_eye_to[0] - r_eye_to[0])))

        # rotate the image
        image_from = Image.fromarray(image_from).rotate(angle)
        image_from = image_to_tensor(image_from).unsqueeze(0)

        #img = np.array(Image.fromarray(image_from).rotate(angle))
        #img = image_to_tensor(img).unsqueeze(0)

        return (image_from, )


"""
def cos_distance(source, test):
    a = np.matmul(np.transpose(source), test)
    b = np.sum(np.multiply(source, source))
    c = np.sum(np.multiply(test, test))
    return np.float64(1 - (a / (np.sqrt(b) * np.sqrt(c))))

def euclidean_distance(source, test, norm=False):
    if norm:
        source = l2_normalize(source)
        test = l2_normalize(test)

    dist = source - test
    dist = np.sum(np.multiply(dist, dist))
    dist = np.sqrt(dist)

    return np.float64(dist)

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))
"""

NODE_CLASS_MAPPINGS = {
    "FaceEmbedDistance": FaceEmbedDistance,
    "FaceAnalysisModels": FaceAnalysisModels,
    "FaceBoundingBox": FaceBoundingBox,
    #"FaceAlign": FaceAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEmbedDistance": "Face Embeds Distance",
    "FaceAnalysisModels": "Face Analysis Models",
    "FaceBoundingBox": "Face Bounding Box",
    #"FaceAlign": "Face Align",
}
