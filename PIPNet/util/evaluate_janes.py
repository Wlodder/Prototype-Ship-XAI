import os, shutil
import argparse
from PIL import Image, ImageDraw as D
import torchvision
from util.func import get_patch_size
from torchvision import transforms
import torch
from util.vis_pipnet import get_img_coordinates
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
    use_opencv = True
except ImportError:
    use_opencv = False
    print("Heatmaps showing where a prototype is found will not be generated because OpenCV is not installed.", flush=True)



from glob import glob
import json

class SuggestionDataset(torch.utils.data.Dataset):

    def __init__(self, image_glob_command, json_glob_command, transform=None):
        self.image_paths = glob(image_glob_command)
        self.json_paths = glob(json_glob_command)
        self.transform = transform
        self.imgs = self.image_paths
        self.classes = os.listdir(os.path.dirname(os.path.dirname(self.image_paths[0])))
        list.sort(self.classes)
        list.sort(self.image_paths)
        list.sort(self.json_paths)
        self.class_to_num = {}
        i = 0
        for cls in self.classes:
            self.class_to_num[cls] = i
            i += 1


    def __len__(self):
        return len(self.image_paths)

    def determine_class(self, path):
        cls = os.path.dirname(path).split('/')[-1]
        return self.class_to_num[cls]

    def class_name(self, path):
        cls = os.path.dirname(path).split('/')[-1]
        return cls

    def get_gt(self, idx):
        j = self.json_paths[idx]
        with open(j,'r')  as f:
            r = json.load(f)

            return r['objects']
    
    def __getitem__(self, idx ):
        img_name =  self.image_paths[idx]
        cls = self.determine_class(img_name)

        image = torchvision.io.read_image(img_name).float()
        image_size = image.size()

        image = self.transform(image)

        return image, cls, self.class_name(img_name), image_size[1], image_size[2]

class ComparisonWrapper():

    def __init__(self, image_name, height, width, cls, json):
        self.image_name = image_name
        self.height = height
        self.width = width
        self.cls = cls
        self.prototypes = []
        self.json = json
        self.top_overlaps = {}

    def add_top_overlaps(self, gt_poly_idx, overlaps):
        if gt_poly_idx not in self.top_overlaps:
            self.top_overlaps[gt_poly_idx] = []
        self.top_overlaps[gt_poly_idx].append(overlaps)

    def add_prototype(self, coordinate):
        self.prototypes.append(coordinate)


    def view_top_overlaps(self, gt_poly_idx):
        print(f"Top overlaps for ground truth polygon {gt_poly_idx}:")
        if gt_poly_idx in self.top_overlaps:
            for overlaps in self.top_overlaps[gt_poly_idx]:
                print(overlaps)
        else:
            print(f"No overlaps found for ground truth polygon {gt_poly_idx}.")

    def __str__(self):
        return f'{self.image_name} {self.cls} {self.height} {self.width}'

def check_prototype_locations(net, device, args: argparse.Namespace):
    # Make sure the model is in evaluation mode
    net.eval()

    save_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images),"Experiments")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    patchsize, skip = get_patch_size(args)

    num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(args.image_size, args.image_size)),
                            # transforms.ToTensor(),
                            normalize])

    # vis_test_set = torchvision.datasets.ImageFolder(imgs_dir, transform=transform_no_augment)

    img_path = os.environ.get('CHECK_JSON_LABEL_PATH_TEST', '') + '/*/*.jpg'
    label_path =  os.environ.get('CHECK_JSON_LABEL_PATH_JSON', '') + '/*/*.json'
    vis_test_set = SuggestionDataset(img_path, label_path, transform_no_augment)
    vis_test_loader = torch.utils.data.DataLoader(vis_test_set, batch_size = 1,
                                                shuffle=False, pin_memory=not args.disable_cuda and torch.cuda.is_available(),
                                                num_workers=num_workers)
    imgs = vis_test_set.imgs

    # Function to calculate IoU between two bounding boxes
    def calculate_iou(box1, box2):
        # Box format is [(x1, y1), (x2, y2)]
        # Calculate intersection area
        x_left = max(box1[0][0], box2[0][0])
        y_top = max(box1[0][1], box2[0][1])
        x_right = min(box1[1][0], box2[1][0])
        y_bottom = min(box1[1][1], box2[1][1])
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of each box
        box1_area = (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1])
        box2_area = (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
        else:
            return intersection_area / union_area

    img_stats = []
    for k, (xs, ys, class_name, h, w) in enumerate(vis_test_loader): #shuffle is false so should lead to same order as in imgs
        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k]
        img_name = os.path.splitext(os.path.basename(img))[0]
        dir = os.path.join(save_dir,img_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
            shutil.copy(img, dir)
        
        img_stats.append(
            ComparisonWrapper(
                img_name, h.item(), w.item(), class_name[0], vis_test_set.get_gt(k)
            )
        )

        with torch.no_grad():
            softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            #sorted_out, sorted_out_indices = torch.sort(out.squeeze(0), descending=True)
            
            save_path = os.path.join(dir, 'prototypes')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            _, sorted_pooled_indices = torch.sort(pooled.squeeze(0), descending=True)
            
            simweights = []

            image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img).convert("RGB"))
            js = vis_test_set.get_gt(k)
            gt_polys = []
            
            # Initialize a dictionary to track prototype overlaps for each gt_poly
            gt_poly_overlaps = []
            
            for label in js:
                poly = label['polygon']

                h_ratio = args.image_size  / h.item()
                w_ratio = args.image_size / w.item()
                min_x, min_y, max_x, max_y = 10000,10000,0,0
                for p in poly:
                    min_x = min(min_x, p['x'])
                    min_y = min(min_y, p['y'])
                    max_x = max(max_x, p['x'])
                    max_y = max(max_y, p['y'])

                gt_poly = [[int(min_x * w_ratio), int(min_y * h_ratio)],[int(max_x * w_ratio),int(max_y * h_ratio)]]
                gt_polys.append(gt_poly)
                
                # Initialize a dictionary to store prototype overlaps for this ground truth polygon
                gt_poly_overlaps.append({})

            # Dictionary to store all prototype activation boxes
            proto_boxes = {}
            
            for prototype_idx in sorted_pooled_indices:
                simweight = pooled[0,prototype_idx].item() 
                if simweight < 0.1:
                    continue
                simweights.append(simweight)

                max_h, max_idx_h = torch.max(softmaxes[0, prototype_idx, :, :], dim=0)
                max_w, max_idx_w = torch.max(max_h, dim=0)
                max_idx_h = max_idx_h[max_idx_w].item()
                max_idx_w = max_idx_w.item()

                
                copy = image.copy()
                draw = D.Draw(copy)
                for gt_poly in gt_polys:
                    draw.rectangle([(gt_poly[0][0],gt_poly[0][1]), (gt_poly[1][0], gt_poly[1][1])], outline='red', width=2)
                    max_idx_h_p, max_idx_w_p = (gt_poly[0][0] + gt_poly[1][0]) // 2 , (gt_poly[0][1] + gt_poly[1][1]) // 2
                    
                    draw.rectangle([(max_idx_h_p - patchsize//2,max_idx_w_p - patchsize//2 ), 
                                    (min(args.image_size, max_idx_h_p+ patchsize//2), min(args.image_size, max_idx_w_p+patchsize//2))], outline='blue', width=2)

                # Draw the prototype activation region
                proto_box = [(max_idx_w*skip,max_idx_h*skip), 
                             (min(args.image_size, max_idx_w*skip+patchsize), min(args.image_size, max_idx_h*skip+patchsize))]
                draw.rectangle(proto_box, outline='yellow', width=2)
                
                # Store the prototype box for later overlap calculations
                proto_boxes[prototype_idx.item()] = {
                    'box': proto_box,
                    'activation': simweight
                }
                
                copy.save(os.path.join(save_path, 'p%s_sim%s_rect.png'%(str(prototype_idx.item()),str(f"{simweight:.3f}"))))

                img_stats[-1].add_prototype(proto_box)
            
            # Calculate overlap between each gt_poly and all prototype boxes
            for i, gt_poly in enumerate(gt_polys):
                for proto_idx, proto_data in proto_boxes.items():

                    max_idx_h_p, max_idx_w_p = (gt_poly[0][0] + gt_poly[1][0]) // 2 , (gt_poly[0][1] + gt_poly[1][1]) // 2
                    
                    poly_box = [(max_idx_h_p - patchsize//2,max_idx_w_p - patchsize//2 ), 
                                    (min(args.image_size, max_idx_h_p+ patchsize//2), min(args.image_size, max_idx_w_p+patchsize//2))]


                    overlap = calculate_iou(gt_poly, proto_data['box'])
                    box_overlap = calculate_iou(poly_box, proto_data['box'])
                    gt_poly_overlaps[i][proto_idx] = {
                        'overlap': overlap,
                        'activation': proto_data['activation'],
                        'box_overlap': box_overlap
                    }
            
            # Find and save the top 10 overlapping prototypes for each gt_poly
            for i, gt_overlaps in enumerate(gt_poly_overlaps):
                # Sort by overlap score
                sorted_overlaps = sorted(gt_overlaps.items(), 
                                        key=lambda x: x[1]['overlap'], 
                                        reverse=True)[:20]
                
                # Create a text file with the top overlaps
                overlap_file_path = os.path.join(save_path, f'gt_poly_{i}_top_prototypes.txt')
                with open(overlap_file_path, 'w') as f:
                    f.write(f"Top 10 prototypes overlapping with ground truth polygon {i}:\n")
                    f.write("-" * 60 + "\n")
                    for j, (proto_idx, data) in enumerate(sorted_overlaps):
                        f.write(f"{j+1}. Prototype {proto_idx}: Overlap={data['overlap']:.4f}, Activation={data['activation']:.4f}\n")
                    f.write('\n')
                    f.write(f"Top 10 prototypes overlapping with ground truth polygon with box {i}:\n")
                    f.write("-" * 60 + "\n")
                    for j, (proto_idx, data) in enumerate(sorted_overlaps):
                        f.write(f"{j+1}. Prototype {proto_idx}: Overlap={data['box_overlap']:.4f}, Activation={data['activation']:.4f}\n")
                
                # Also create a visualization showing the top overlapping prototypes
                vis_img = image.copy()
                draw = D.Draw(vis_img)
                
                # Draw the ground truth polygon in red
                draw.rectangle([(gt_polys[i][0][0], gt_polys[i][0][1]), 
                               (gt_polys[i][1][0], gt_polys[i][1][1])], 
                              outline='red', width=3)
                
                # Draw the top 3 prototype boxes with different colors
                colors = ['yellow', 'green', 'blue', 'purple', 'cyan']
                for j, (proto_idx, data) in enumerate(sorted_overlaps[:5]):  # Show top 5 with different colors
                    proto_box = proto_boxes[proto_idx]['box']
                    color = colors[j % len(colors)]
                    draw.rectangle(proto_box, outline=color, width=2)
                    # Add prototype number for reference
                    draw.text((proto_box[0][0], proto_box[0][1] - 15), 
                             f"P{proto_idx} ({data['overlap']:.2f})", 
                             fill=color)
                
                # Save the visualization
                vis_img.save(os.path.join(save_path, f'gt_poly_{i}_top_overlaps.png'))
                
                # Add the top overlaps to img_stats if ComparisonWrapper supports it
                if hasattr(img_stats[-1], 'add_top_overlaps'):
                    img_stats[-1].add_top_overlaps(i, sorted_overlaps)
    
    return img_stats