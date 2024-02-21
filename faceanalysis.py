import dlib
import torch
import torchvision.transforms.v2 as T
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor

DLIB_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlib")

class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("ANALYSIS_MODELS", )
    FUNCTION = "load_models"
    CATEGORY = "FaceAnalysis"

    def load_models(self):
        return ({
            "detector": dlib.get_frontal_face_detector(),
            "shape_predict": dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat")),
            "face_recog": dlib.face_recognition_model_v1(os.path.join(DLIB_DIR, "dlib_face_recognition_resnet_model_v1.dat")),
            }, )

class FaceEmbedDistance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "reference": ("IMAGE", ),
                "image": ("IMAGE", ),
                "generate_image_overlay": ("BOOLEAN", { "default": True })
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE", "euclidean", "cosine", "csv")
    OUTPUT_NODE = True
    FUNCTION = "analize"
    CATEGORY = "FaceAnalysis"

    def analize(self, analysis_models, reference, image, generate_image_overlay=True):
        if generate_image_overlay:
            font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"), 32)
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        self.detector = analysis_models.get("detector")
        self.shape_predict = analysis_models.get("shape_predict")
        self.face_recog = analysis_models.get("face_recog")

        ref = np.array(T.ToPILImage()(reference[0].permute(2, 0, 1)).convert('RGB'))
        ref = self.get_descriptor(ref)
        if ref is None:
            raise Exception('No face detected in reference image')

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
                if ref == img: # Same face
                    eucl_dist = 0.0
                    cos_dist = 0.0
                else:
                    eucl_dist = np.linalg.norm(np.array(ref) - np.array(img))
                    cos_dist = 1 - np.dot(ref, img) / (np.linalg.norm(ref) * np.linalg.norm(img))
            
            out_eucl.append(eucl_dist)
            out_cos.append(cos_dist)
            
            print(f"\033[96mFace Analysis: Euclidean: {eucl_dist}, Cosine: {cos_dist}\033[0m")

            eucl_dist = round(eucl_dist, 3)
            cos_dist = round(cos_dist, 3)

            if generate_image_overlay:
                tmp = T.ToPILImage()(i.permute(2, 0, 1)).convert('RGBA')
                txt = Image.new('RGBA', (image.shape[2], txt_height), color=background_color)
                draw = ImageDraw.Draw(txt)
                draw.text((0, 0), f"EUC: {eucl_dist} | COS-1: {cos_dist}", font=font, fill=(255, 255, 255, 255))
                composite = Image.new('RGBA', tmp.size)
                composite.paste(txt, (0, tmp.height - txt.height))
                composite = Image.alpha_composite(tmp, composite)
                out.append(T.ToTensor()(composite))

        if out:        
            img = torch.stack(out).permute(0, 2, 3, 1)
        else:
            img = image
        
        csv = "id,euclidean,cosine\n"
        if len(out_eucl) == 1:
            out_eucl = out_eucl[0]
            out_cos = out_cos[0]
            csv += f"0,{out_eucl},{out_cos}\n"
        else:
            for id, (eucl, cos) in enumerate(zip(out_eucl, out_cos)):
                csv += f"{id},{eucl},{cos}\n"

        return(img, out_eucl, out_cos, csv,)
    
    def get_descriptor(self, image):
        faces = self.detector(image)
        if len(faces) > 0:
            shape = self.shape_predict(image, faces[0])
            return self.face_recog.compute_face_descriptor(image, shape)

        return None

NODE_CLASS_MAPPINGS = {
    "FaceEmbedDistance": FaceEmbedDistance,
    "FaceAnalysisModels": FaceAnalysisModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEmbedDistance": "Face Embeds Distance",
    "FaceAnalysisModels": "Face Analysis Models",
}
