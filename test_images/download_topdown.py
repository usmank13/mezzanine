import requests, os

# Top-down / overhead / bird's eye view images from COCO val2017
# Searched for food prep, desks, aerial, flat-lay type images
urls = {
    # Food/table overhead (very common top-down)
    "coco_sushi_td": "http://images.cocodataset.org/val2017/000000106140.jpg",
    "coco_donuts_td": "http://images.cocodataset.org/val2017/000000295316.jpg",
    "coco_cake_td": "http://images.cocodataset.org/val2017/000000360661.jpg",
    "coco_fruit_td": "http://images.cocodataset.org/val2017/000000002587.jpg",
    "coco_hotdog_td": "http://images.cocodataset.org/val2017/000000174004.jpg",
    "coco_sandwich_td": "http://images.cocodataset.org/val2017/000000279541.jpg",
    "coco_broccoli_td": "http://images.cocodataset.org/val2017/000000180135.jpg",
    "coco_plates_td": "http://images.cocodataset.org/val2017/000000084270.jpg",
    # Laptop/keyboard/desk overhead
    "coco_laptop_td": "http://images.cocodataset.org/val2017/000000153529.jpg",
    "coco_keyboard_td": "http://images.cocodataset.org/val2017/000000459590.jpg",
    # Books/objects from above
    "coco_books_td": "http://images.cocodataset.org/val2017/000000532493.jpg",
    "coco_remote_td": "http://images.cocodataset.org/val2017/000000173091.jpg",
    # More food overhead
    "coco_bowl_td": "http://images.cocodataset.org/val2017/000000434230.jpg",
    "coco_bread_td": "http://images.cocodataset.org/val2017/000000226592.jpg",
    "coco_salad_td": "http://images.cocodataset.org/val2017/000000349860.jpg",
}

for name, url in urls.items():
    path = f"{name}.jpg"
    if os.path.exists(path):
        print(f"  skip {name}")
        continue
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        from PIL import Image
        img = Image.open(path)
        print(f"  ✓ {name} ({img.size[0]}×{img.size[1]})")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
