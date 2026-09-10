from ultralytics import YOLO

model = YOLO('/home/fang/Downloads/HackSphere/weights/wasteland_model/WtE_Predictor/v3_final_refinement/weights/best.pt') 

def get_waste_composition_from_results(result, class_names):
    img_h, img_w = result.orig_shape
    total_img_area = img_h * img_w

    raw_areas = {name: 0.0 for name in class_names.values()}

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]

        w, h = box.xywh[0][2], box.xywh[0][3]
        box_area = w * h
        raw_areas[cls_name] += (box_area.item() / total_img_area) * 100

    total_detected = sum(raw_areas.values())
    if total_detected > 100:
        factor = 100 / total_detected
        return {k: v * factor for k, v in raw_areas.items()}

    return raw_areas

def get_waste_composition(img_path):

    results = model.predict(source=img_path, conf=0.0001)
    return get_waste_composition_from_results(results[0], model.names)