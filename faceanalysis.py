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
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from comfy.utils import ProgressBar
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

def expand_mask(mask, expand, tapered_corners):
    import scipy

    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    device = mask.device
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)

    return torch.stack(out, dim=0).to(device)

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, 
                                 c2.T - (s2 / s1) * R * c1.T)),
                                 np.matrix([0., 0., 1.])])

def mask_from_landmarks(image, landmarks):
    import cv2

    mask = np.zeros(image.shape[:2], dtype=np.float64)
    points = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, points, color=1)

    return mask

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
    
    def get_keypoints(self, image):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]['kps']
            right_eye = shape[0]
            left_eye = shape[1]
            nose = shape[2]
            left_mouth = shape[3]
            right_mouth = shape[4]
            
            return [left_eye, right_eye, nose, left_mouth, right_mouth]
        return None

    def get_landmarks(self, image, extended_landmarks=False):
        face = self.get_face(image)
        if face is not None:
            shape = face[0]['landmark_2d_106']
            landmarks = np.round(shape).astype(np.int64)

            main_features = landmarks[33:]
            left_eye = landmarks[87:97]
            right_eye = landmarks[33:43]
            eyes = landmarks[[*range(33,43), *range(87,97)]]
            nose = landmarks[72:87]
            mouth = landmarks[52:72]
            left_brow = landmarks[97:106]
            right_brow = landmarks[43:52]
            outline = landmarks[[*range(33), *range(48,51), *range(102, 105)]]
            outline_forehead = outline

            return [landmarks, main_features, eyes, left_eye, right_eye, nose, mouth, left_brow, right_brow, outline, outline_forehead]
        return None

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
        #faces, scores, _ = self.face_detector.run(np.array(image), 1, -1)
        
        if len(faces) > 0:
            return sorted(faces, key=lambda x: x.area(), reverse=True)
            #return [face for _, face in sorted(zip(scores, faces), key=lambda x: x[0], reverse=True)] # sort by score
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
    
    def get_keypoints(self, image):
        faces = self.get_face(image)
        if faces is not None:
            shape = self.shape_predictor(image, faces[0])
          
            left_eye = [(shape.part(0).x + shape.part(1).x // 2), (shape.part(0).y + shape.part(1).y) // 2]
            right_eye = [(shape.part(2).x + shape.part(3).x // 2), (shape.part(2).y + shape.part(3).y) // 2]
            nose = [shape.part(4).x, shape.part(4).y]

            return [left_eye, right_eye, nose]
        return None
    
    def get_landmarks(self, image, extended_landmarks=False):
        if extended_landmarks:
            if not os.path.exists(os.path.join(DLIB_DIR, "shape_predictor_81_face_landmarks.dat")):
                raise Exception("The 68 point landmark model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/shape_predictor_81_face_landmarks.dat")
            predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_81_face_landmarks.dat"))
        else:
            if not os.path.exists(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat")):
                raise Exception("The 68 point landmark model is not available. Please download it from https://huggingface.co/matt3ounstable/dlib_predictor_recognition/blob/main/shape_predictor_68_face_landmarks.dat")
            predictor = dlib.shape_predictor(os.path.join(DLIB_DIR, "shape_predictor_68_face_landmarks.dat"))

        faces = self.get_face(image)
        if faces is not None:
            shape = predictor(image, faces[0])
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            main_features = landmarks[17:68]
            left_eye = landmarks[42:48]
            right_eye = landmarks[36:42]
            eyes = landmarks[36:48]
            nose = landmarks[27:36]
            mouth = landmarks[48:68]
            left_brow = landmarks[17:22]
            right_brow = landmarks[22:27]
            outline = landmarks[[*range(17), *range(26,16,-1)]]
            if extended_landmarks:
                outline_forehead = landmarks[[*range(17), *range(26,16,-1), *range(68, 81)]]
            else:
                outline_forehead = outline

            return [landmarks, main_features, eyes, left_eye, right_eye, nose, mouth, left_brow, right_brow, outline, outline_forehead]
        return None


class FaceAnalysisModels:
    @classmethod
    def INPUT_TYPES(s):
        libraries = []
        if IS_INSIGHTFACE_INSTALLED:
            libraries.append("insightface")
            libraries.append("auraface")
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
            out = InsightFace(provider=provider)
        elif library == "auraface":
            out = InsightFace(provider=provider, name="auraface")
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
                    norm_dist = 0.0
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
    
        out = torch.stack(out)

        # filter out the best matches
        if filter_best > 0:
            filter_best = min(filter_best, len(out))
            out_dist, idx = torch.topk(torch.tensor(out_dist), filter_best, largest=False)
            out = out[idx]
            out_dist = out_dist.cpu().numpy().tolist()
        
        if out.shape[3] > 3:
            out = out[:, :, :, :3]

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
        shape = analysis_models.get_keypoints(image_from)
        l_eye_from = shape[0]
        r_eye_from = shape[1]
        angle = float(np.degrees(np.arctan2(l_eye_from[1] - r_eye_from[1], l_eye_from[0] - r_eye_from[0])))

        if image_to is not None:
            image_to = tensor_to_image(image_to[0])
            shape = analysis_models.get_keypoints(image_to)
            l_eye_to = shape[0]
            r_eye_to = shape[1]
            angle -= float(np.degrees(np.arctan2(l_eye_to[1] - r_eye_to[1], l_eye_to[0] - r_eye_to[0])))

        # rotate the image
        image_from = Image.fromarray(image_from).rotate(angle)
        image_from = image_to_tensor(image_from).unsqueeze(0)

        #img = np.array(Image.fromarray(image_from).rotate(angle))
        #img = image_to_tensor(img).unsqueeze(0)

        return (image_from, )

class faceSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image": ("IMAGE", ),
                "area": (["face", "main_features", "eyes", "left_eye", "right_eye", "nose", "mouth", "face+forehead (if available)"], ),
                "grow": ("INT", { "default": 0, "min": -4096, "max": 4096, "step": 1 }),
                "grow_tapered": ("BOOLEAN", { "default": False }),
                "blur": ("INT", { "default": 13, "min": 1, "max": 4096, "step": 2 }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "image", "seg_mask", "seg_image", "x", "y", "width", "height")
    FUNCTION = "segment"
    CATEGORY = "FaceAnalysis"

    def segment(self, analysis_models, image, area, grow, grow_tapered, blur):
        steps = image.shape[0]

        if steps > 1:
            pbar = ProgressBar(steps)

        out_mask = []
        out_image = []
        out_seg_mask = []
        out_seg_image = []
        out_x = []
        out_y = []
        out_w = []
        out_h = []

        for img in image:       
            face = tensor_to_image(img)

            if face is None:
                print(f"\033[96mNo face detected at frame {len(out_image)}\033[0m")
                img = torch.zeros_like(img)
                mask = img.clone()[:,:,:1]
                out_mask.append(mask)
                out_image.append(img)
                out_seg_mask.append(mask[:8,:8,:])
                out_seg_image.append(img[:8,:8,:])
                out_x.append(0)
                out_y.append(0)
                continue

            landmarks = analysis_models.get_landmarks(face, extended_landmarks=("forehead" in area))

            if landmarks is None:
                print(f"\033[96mNo landmarks detected at frame {len(out_image)}\033[0m")
                img = torch.zeros_like(img)
                mask = img.clone()[:,:,:1]
                out_mask.append(mask)
                out_image.append(img)
                out_seg_mask.append(mask[:8,:8,:])
                out_seg_image.append(img[:8,:8,:])
                out_x.append(0)
                out_y.append(0)
                continue

            if area == "face":
                landmarks = landmarks[-2]
            elif area == "eyes":
                landmarks = landmarks[2]
            elif area == "left_eye":
                landmarks = landmarks[3]
            elif area == "right_eye":
                landmarks = landmarks[4]
            elif area == "nose":
                landmarks = landmarks[5]
            elif area == "mouth":
                landmarks = landmarks[6]
            elif area == "main_features":
                landmarks = landmarks[1]
            elif "forehead" in area:
                landmarks = landmarks[-1]

            #mask = np.zeros(face.shape[:2], dtype=np.float64)
            #points = cv2.convexHull(landmarks)
            #cv2.fillConvexPoly(mask, points, color=1)

            mask = mask_from_landmarks(face, landmarks)
            mask = image_to_tensor(mask).unsqueeze(0).squeeze(-1).clamp(0, 1).to(device=img.device)

            _, y, x = torch.where(mask)
            x1, x2 = x.min().item(), x.max().item()
            y1, y2 = y.min().item(), y.max().item()
            smooth = int(min(max((x2 - x1), (y2 - y1)) * 0.2, 99))

            if smooth > 1:
                if smooth % 2 == 0:
                    smooth+= 1
                mask = T.functional.gaussian_blur(mask.bool().unsqueeze(1), smooth).squeeze(1).float()
            
            if grow != 0:
                mask = expand_mask(mask, grow, grow_tapered)

            if blur > 1:
                if blur % 2 == 0:
                    blur+= 1
                mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1).float()

            mask = mask.squeeze(0).unsqueeze(-1)

            # extract segment from image
            y, x, _ = torch.where(mask)
            x1, x2 = x.min().item(), x.max().item()
            y1, y2 = y.min().item(), y.max().item()
            segment_mask = mask[y1:y2, x1:x2, :]
            segment_image = img[y1:y2, x1:x2, :]
            
            img = img * mask.repeat(1, 1, 3)

            out_mask.append(mask)
            out_image.append(img)
            out_seg_mask.append(segment_mask)
            out_seg_image.append(segment_image)
            out_x.append(x1)
            out_y.append(y1)

            if steps > 1:
                pbar.update(1)
        
        out_mask = torch.stack(out_mask).squeeze(-1)
        out_image = torch.stack(out_image)

        # find the max size of out_seg_image
        max_w = max([img.shape[1] for img in out_seg_image])
        max_h = max([img.shape[0] for img in out_seg_image])
        pad_left = [(max_w - img.shape[1])//2 for img in out_seg_image]
        pad_right = [max_w - img.shape[1] - pad_left[i] for i, img in enumerate(out_seg_image)]
        pad_top = [(max_h - img.shape[0])//2 for img in out_seg_image]
        pad_bottom = [max_h - img.shape[0] - pad_top[i] for i, img in enumerate(out_seg_image)]
        out_seg_image = [F.pad(img.unsqueeze(0).permute([0,3,1,2]), (pad_left[i], pad_right[i], pad_top[i], pad_bottom[i])) for i, img in enumerate(out_seg_image)]
        out_seg_mask = [F.pad(mask.unsqueeze(0).permute([0,3,1,2]), (pad_left[i], pad_right[i], pad_top[i], pad_bottom[i])) for i, mask in enumerate(out_seg_mask)]

        out_seg_image = torch.cat(out_seg_image).permute([0,2,3,1])
        out_seg_mask = torch.cat(out_seg_mask).squeeze(1)

        if len(out_x) == 1:
            out_x = out_x[0]
            out_y = out_y[0]

        out_w = out_seg_image.shape[2]
        out_h = out_seg_image.shape[1]

        return (out_mask, out_image, out_seg_mask, out_seg_image, out_x, out_y, out_w, out_h)


class FaceWarp:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "analysis_models": ("ANALYSIS_MODELS", ),
                "image_from": ("IMAGE", ),
                "image_to": ("IMAGE", ),
                "keypoints": (["main features", "full face", "full face+forehead (if available)"], ),
                "grow": ("INT", { "default": 0, "min": -4096, "max": 4096, "step": 1 }),
                "blur": ("INT", { "default": 13, "min": 1, "max": 4096, "step": 2 }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "warp"
    CATEGORY = "FaceAnalysis"

    def warp(self, analysis_models, image_from, image_to, keypoints, grow, blur):
        import cv2
        from color_matcher import ColorMatcher
        from color_matcher.normalizer import Normalizer

        if image_from.shape[0] < image_to.shape[0]:
            image_from = torch.cat([image_from, image_from[-1:].repeat((image_to.shape[0]-image_from.shape[0], 1, 1, 1))], dim=0)
        elif image_from.shape[0] > image_to.shape[0]:
            image_from = image_from[:image_to.shape[0]]

        steps = image_from.shape[0]
        if steps > 1:
            pbar = ProgressBar(steps)

        cm = ColorMatcher()

        result_image = []
        result_mask = []

        for i in range(steps):
            img_from = tensor_to_image(image_from[i])
            img_to = tensor_to_image(image_to[i])

            shape_from = analysis_models.get_landmarks(img_from, extended_landmarks=("forehead" in keypoints))
            shape_to = analysis_models.get_landmarks(img_to, extended_landmarks=("forehead" in keypoints))

            if shape_from is None or shape_to is None:
                print(f"\033[96mNo landmarks detected at frame {i}\033[0m")
                img = image_to[i].unsqueeze(0)
                mask = torch.zeros_like(img)[:,:,:1]
                result_image.append(img)
                result_mask.append(mask)
                continue

            if keypoints == "main features":
                shape_from = shape_from[1]
                shape_to = shape_to[1]
            else:
                shape_from = shape_from[0]
                shape_to = shape_to[0]

            # get the transformation matrix
            from_points = np.array(shape_from, dtype=np.float64)
            to_points = np.array(shape_to, dtype=np.float64)
            
            matrix = cv2.estimateAffine2D(from_points, to_points)[0]
            output = cv2.warpAffine(img_from, matrix, (img_to.shape[1], img_to.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            mask_from = mask_from_landmarks(img_from, shape_from)
            mask_to = mask_from_landmarks(img_to, shape_to)
            output_mask = cv2.warpAffine(mask_from, matrix, (img_to.shape[1], img_to.shape[0]))

            output_mask = torch.from_numpy(output_mask).unsqueeze(0).unsqueeze(-1).float()
            mask_to = torch.from_numpy(mask_to).unsqueeze(0).unsqueeze(-1).float()
            output_mask = torch.min(output_mask, mask_to)

            output = image_to_tensor(output).unsqueeze(0)
            img_to = image_to_tensor(img_to).unsqueeze(0)
            
            if grow != 0:
                output_mask = expand_mask(output_mask.squeeze(-1), grow, True).unsqueeze(-1)

            if blur > 1:
                if blur % 2 == 0:
                    blur+= 1
                output_mask = T.functional.gaussian_blur(output_mask.permute(0,3,1,2), blur).permute(0,2,3,1)

            padding = 0

            _, y, x, _ = torch.where(mask_to)
            x1 = max(0, x.min().item() - padding)
            y1 = max(0, y.min().item() - padding)
            x2 = min(img_to.shape[2], x.max().item() + padding)
            y2 = min(img_to.shape[1], y.max().item() + padding)
            cm_ref = img_to[:, y1:y2, x1:x2, :]

            _, y, x, _ = torch.where(output_mask)
            x1 = max(0, x.min().item() - padding)
            y1 = max(0, y.min().item() - padding)
            x2 = min(output.shape[2], x.max().item() + padding)
            y2 = min(output.shape[1], y.max().item() + padding)
            cm_image = output[:, y1:y2, x1:x2, :]

            normalized = cm.transfer(src=Normalizer(cm_image[0].numpy()).type_norm() , ref=Normalizer(cm_ref[0].numpy()).type_norm(), method='mkl')
            normalized = torch.from_numpy(normalized).unsqueeze(0)

            factor = 0.8

            output[:, y1:y1+cm_image.shape[1], x1:x1+cm_image.shape[2], :] = factor * normalized + (1 - factor) * cm_image

            output_image = output * output_mask + img_to * (1 - output_mask)
            output_image = output_image.clamp(0, 1)
            output_mask = output_mask.clamp(0, 1).squeeze(-1)

            result_image.append(output_image)
            result_mask.append(output_mask)

            if steps > 1:
                pbar.update(1)
        
        result_image = torch.cat(result_image, dim=0)
        result_mask = torch.cat(result_mask, dim=0)

        return (result_image, result_mask)

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
    "FaceAlign": FaceAlign,
    "FaceSegmentation": faceSegmentation,
    "FaceWarp": FaceWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceEmbedDistance": "Face Embeds Distance",
    "FaceAnalysisModels": "Face Analysis Models",
    "FaceBoundingBox": "Face Bounding Box",
    "FaceAlign": "Face Align",
    "FaceSegmentation": "Face Segmentation",
    "FaceWarp": "Face Warp",
}
