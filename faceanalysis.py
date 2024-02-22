import dlib
from insightface.app import FaceAnalysis
import torch
import torchvision.transforms.v2 as T
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
            "library": (["dlib", "insightface"], ),
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
                "generate_image_overlay": ("BOOLEAN", { "default": True })
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("IMAGE", "euclidean", "cosine", "csv")
    OUTPUT_NODE = True
    FUNCTION = "analize"
    CATEGORY = "FaceAnalysis"

    def analize(self, analysis_models, reference, image, filter_thresh_eucl=1.0, filter_thresh_cos=1.0, generate_image_overlay=True):
        if generate_image_overlay:
            font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Inconsolata.otf"), 32)
            background_color = ImageColor.getrgb("#000000AA")
            txt_height = font.getmask("Q").getbbox()[3] + font.getmetrics()[1]

        self.analysis_models = analysis_models

        #if reference.shape[0] > 1:
        #    reference = torch.mean(reference, dim=0).unsqueeze(0)

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

        img = torch.stack(out)
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEmbedDistance": "Face Embeds Distance",
    "FaceAnalysisModels": "Face Analysis Models",
}
