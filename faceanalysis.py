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

INSTALLED_LIBRARIES = []
if IS_DLIB_INSTALLED:
    INSTALLED_LIBRARIES.append("dlib")
if IS_INSIGHTFACE_INSTALLED:
    INSTALLED_LIBRARIES.append("insightface")

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import comfy.utils
import os
import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

DLIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlib")
INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "library": (INSTALLED_LIBRARIES, ),
            "provider": (["CPU", "CUDA", "DirectML", "OpenVINO", "ROCM", "CoreML"], ),
        }}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "FaceAnalysis"

    def load_models(self, library, provider):
        out = {}

        if library == "insightface":
            out = {
                "library": library,
                "detector": FaceAnalysis(name="buffalo_l", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',])
            }
            out["detector"].prepare(ctx_id=0, det_size=(640, 640))
        else:
            out = {
                "library": library,
                "detector": dlib.get_frontal_face_detector(),
                "shape_predict": dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat")),
                "face_recog": dlib.face_recognition_model_v1(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat")),
            }
        

        return (out, )

def crop_face(image, x, y, w, h, padding=0):
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.width, w + 2 * padding)
    h = min(image.height, h + 2 * padding)

    return image.crop((x, y, x + w, y + h))

class FaceBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "index": ("INT", { "default": -1, "min": -1, "max": 4096, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "x", "y", "width", "height")
    FUNCTION = "bbox"
    CATEGORY = "FaceAnalysis"
    OUTPUT_IS_LIST = (True, True, True, True, True,)

    def bbox(self, analysis_models, image, padding, index=-1):
        out_img = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []

        for i in image:
            img = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')

            if analysis_models["library"] == "insightface":
                faces = analysis_models["detector"].get(np.array(img))
                for face in faces:
                    x, y, w, h = face.bbox.astype(int)
                    w = w - x
                    h = h - y
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img.width, w + 2 * padding)
                    h = min(img.height, h + 2 * padding)
                    crop = img.crop((x, y, x + w, y + h))
                    out_img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
                    out_x.append(x)
                    out_y.append(y)
                    out_w.append(w)
                    out_h.append(h)

            else:
                faces = analysis_models["detector"](np.array(img), 1)
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img.width, w + 2 * padding)
                    h = min(img.height, h + 2 * padding)
                    crop = img.crop((x, y, x + w, y + h))
                    out_img.append(T.ToTensor()(crop).permute(1, 2, 0).unsqueeze(0))
                    out_x.append(x)
                    out_y.append(y)
                    out_w.append(w)
                    out_h.append(h)

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
                "filter_thresh_eucl": ("FLOAT", { "default": 1.0, "min": 0.001, "max": 2.0, "step": 0.001 }),
                "filter_thresh_cos": ("FLOAT", { "default": 1.0, "min": 0.001, "max": 2.0, "step": 0.001 }),
                "filter_best": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1 }),
                "generate_image_overlay": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "euclidean", "cosine")
    OUTPUT_IS_LIST = (False, True, True)
    FUNCTION = "analize"
    CATEGORY = "FaceAnalysis"

    def analize(self, analysis_models, reference, image, filter_thresh_eucl=1.0, filter_thresh_cos=1.0, filter_best=0, generate_image_overlay=True):
        if generate_image_overlay:
            font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"), 32)
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        self.analysis_models = analysis_models

        ref = []
        for i in reference:
            ref_emb = self.get_descriptor(np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB')))
            if ref_emb is not None:
                ref.append(torch.from_numpy(ref_emb))
        
        if ref == []:
            raise Exception('No face detected in reference image')

        ref = torch.stack(ref)
        ref = np.array(torch.mean(ref, dim=0))

        out = []
        out_eucl = []
        out_cos = []
        
        for i in image:
            img = np.array(T.ToPILImage()(i.permute(2, 0, 1)).convert('RGB'))

            img = self.get_descriptor(img)

            if img is None: # No face detected
                eucl_dist = 1.0
                cos_dist = 1.0
            else:
                if np.array_equal(ref, img): # Same face
                    eucl_dist = 0.0
                    cos_dist = 0.0
                else:
                    eucl_dist = np.float64(np.linalg.norm(ref - img))
                    cos_dist = 1 - np.dot(ref, img) / (np.linalg.norm(ref) * np.linalg.norm(img))
            
            if eucl_dist <= filter_thresh_eucl and cos_dist <= filter_thresh_cos:
                print(f"\033[96mFace Analysis: Euclidean: {eucl_dist}, Cosine: {cos_dist}\033[0m")

                if generate_image_overlay:
                    tmp = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGBA')
                    txt = Image.new('RGBA', (image.shape[2], txt_height), color=background_color)
                    draw = ImageDraw.Draw(txt)
                    draw.text((0, 0), f"EUC: {round(eucl_dist, 3)} | COS: {round(cos_dist, 3)}", font=font, fill=(255, 255, 255, 255))
                    composite = Image.new('RGBA', tmp.size)
                    composite.paste(txt, (0, tmp.height - txt.height))
                    composite = Image.alpha_composite(tmp, composite)
                    out.append(T.ToTensor()(composite).permute(1, 2, 0))
                else:
                    out.append(i)

                out_eucl.append(eucl_dist)
                out_cos.append(cos_dist)

        if not out:
            raise Exception('No image matches the filter criteria.')

        # filter out the best matches
        if filter_best > 0:
            out = np.array(out)
            out_eucl = np.array(out_eucl)
            out_cos = np.array(out_cos)
            idx = np.argsort((out_eucl + out_cos) / 2)
            out = torch.from_numpy(out[idx][:filter_best])
            out_eucl = out_eucl[idx][:filter_best].tolist()
            out_cos = out_cos[idx][:filter_best].tolist()

        if isinstance(out, list):
            out = torch.stack(out)

        return(out, out_eucl, out_cos,)
    
    def get_descriptor(self, image):
        embeds = None

        if self.analysis_models["library"] == "insightface":
            faces = self.analysis_models["detector"].get(image)
            if len(faces) > 0:
                embeds = faces[0].normed_embedding
        else:
            faces = self.analysis_models["detector"](image)
            if len(faces) > 0:
                shape = self.analysis_models["shape_predict"](image, faces[0])
                embeds = np.array(self.analysis_models["face_recog"].compute_face_descriptor(image, shape))

        return embeds

NODE_CLASS_MAPPINGS = {
    "FaceEmbedDistance": FaceEmbedDistance,
    "FaceAnalysisModels": FaceAnalysisModels,
    "FaceBoundingBox": FaceBoundingBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEmbedDistance": "Face Embeds Distance",
    "FaceAnalysisModels": "Face Analysis Models",
    "FaceBoundingBox": "Face Bounding Box",
}
