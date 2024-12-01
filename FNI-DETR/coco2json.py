import json

def convert_predictions(pred_json_path, anno_json_path, output_json_path):
    with open(anno_json_path, 'r') as f:
        coco_data = json.load(f)

    file_name_to_id = {img['file_name']: img['id'] for img in coco_data['images']}

    with open(pred_json_path, 'r') as f:
        predictions = json.load(f)

    for pred in predictions:
        file_name = pred['image_id'] + ".jpg"

        if file_name in file_name_to_id:
            pred['image_id'] = file_name_to_id[file_name]  
        else:
            print(f"Warning: {file_name} not found in COCO annotations")

    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Conversion complete! Saved to {output_json_path}")


convert_predictions(
    pred_json_path='runs/val/detr-mb2/predictions.json',
    anno_json_path='/root/autodl-tmp/coco/annotations/instances_val2017.json',
    output_json_path='runs/val/detr-mb2/new_predictions.json'
)
