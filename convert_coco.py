import json
import os


def convert_coco_json(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    images = {
        img["id"]: (img["width"], img["height"], img["file_name"])
        for img in data.get("images", [])
    }

    img_annotations = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    converted_count = 0

    for img_id, (width, height, file_name) in images.items():
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        txt_name = f"{base_name}.txt"
        txt_path = os.path.join(output_dir, txt_name)

        lines = []
        if img_id in img_annotations:
            for ann in img_annotations[img_id]:
                category_id = ann["category_id"] - 1

                bbox = ann["bbox"]
                x_center = (bbox[0] + bbox[2] / 2) / width
                y_center = (bbox[1] + bbox[3] / 2) / height
                w = bbox[2] / width
                h = bbox[3] / height

                kpts = ann.get("keypoints", [])
                kpts_str = []
                for i in range(0, len(kpts), 3):
                    kx = kpts[i] / width
                    ky = kpts[i + 1] / height
                    vis = kpts[i + 2]
                    v_flag = 2 if vis > 0 else 0
                    kpts_str.append(f"{kx:.6f} {ky:.6f} {v_flag}")

                line = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} " + " ".join(
                    kpts_str
                )
                lines.append(line)

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        converted_count += 1

    return converted_count


if __name__ == "__main__":
    # Mengarah langsung ke dalam folder 'dataset'
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    splits = ["train", "valid", "test"]

    print("=== MENGOLAH KONVERSI COCO KE YOLO POSE ===")

    for split in splits:
        json_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
        output_dir = os.path.join(dataset_dir, split, "labels")

        if os.path.exists(json_path):
            print(f"\n[+] Memproses folder 'dataset/{split}'...")
            total = convert_coco_json(json_path, output_dir)
            print(
                f"    Berhasil mengonversi {total} file ke: {os.path.relpath(output_dir)}"
            )
        else:
            print(
                f"\n[-] Skipping 'dataset/{split}': File tidak ditemukan di"
            )
            print(f"    Path: {json_path}")

    print("\n=== PROSES SELESAI ===")