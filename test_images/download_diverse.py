import requests, os

# More diverse viewpoints for warrant gap analysis
urls = {
    # Top-down / overhead
    "coco_desk_topdown": "http://images.cocodataset.org/val2017/000000056350.jpg",
    "coco_skateboard": "http://images.cocodataset.org/val2017/000000087038.jpg",
    "coco_street_overhead": "http://images.cocodataset.org/val2017/000000176446.jpg",
    # Side/lateral views
    "coco_bus_side": "http://images.cocodataset.org/val2017/000000087470.jpg",
    "coco_train_side": "http://images.cocodataset.org/val2017/000000131131.jpg",
    # Indoor scenes
    "coco_bathroom": "http://images.cocodataset.org/val2017/000000181859.jpg",
    "coco_living_room": "http://images.cocodataset.org/val2017/000000087875.jpg",
    # Outdoor/landscape
    "coco_beach": "http://images.cocodataset.org/val2017/000000200365.jpg",
    "coco_mountain": "http://images.cocodataset.org/val2017/000000226111.jpg",
    # Close-up / macro-ish
    "coco_clock": "http://images.cocodataset.org/val2017/000000348881.jpg",
    "coco_bear": "http://images.cocodataset.org/val2017/000000459467.jpg",
    # Unusual angles
    "coco_giraffe_up": "http://images.cocodataset.org/val2017/000000579635.jpg",
    "coco_boat": "http://images.cocodataset.org/val2017/000000284991.jpg",
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
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
