import torch
from PIL import Image
import torchvision.transforms as T
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='images-finerva-price')
    parser.add_argument('--output_dir', type=str, default='vision_features-fin-price')
    parser.add_argument('--img_type', type=str, default="vit", choices=['detr', 'vit'], help='type of image features')
    args = parser.parse_args()
    return args

def extract_features(img_type, input_image):
    if img_type == "vit":
        config = resolve_data_config({}, model=vit_model)
        transform = create_transform(**config)
        with torch.no_grad():
            img = Image.open(input_image).convert("RGB")
            input = transform(img).unsqueeze(0)
            device = next(vit_model.parameters()).device
            input = input.to(device)
            feature = vit_model.forward_features(input)
        return feature
    
    elif img_type == "detr":
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            img = Image.open(input_image).convert("RGB")
            input = transform(img).unsqueeze(0)
            feature = detr_model(input)[-1]
        return feature

if __name__ == '__main__':
    args = parse_args()
    print("args",args)
    all_images = os.listdir(args.data_root)
    tmp = []
    name_map = {}
    print(len(all_images))
    if args.img_type == "vit":


        vit_model = timm.create_model(
            "vit_large_patch32_384",
            pretrained=False,    
            num_classes=0       
        ).cuda()
        vit_model.eval()
        def forward_features_patch(x):
            x = vit_model.patch_embed(x)                                 # -> [B, num_patches, C]
            cls_token = vit_model.cls_token.expand(x.size(0), -1, -1)    # -> [B, 1, C]
            x = torch.cat((cls_token, x), dim=1)                         # -> [B, num_patches+1, C]
            x = x + vit_model.pos_embed                                  
            x = vit_model.pos_drop(x)
            for blk in vit_model.blocks:
                x = blk(x)
            x = vit_model.norm(x)                                       
            return x                                                     # -> [B, seq_len, C]
        # 打补丁
        vit_model.forward_features = forward_features_patch

        state_dict = torch.load(
            "checkpoints/jx_vit_large_p32_384-9b920ba8.pth",
            map_location="cuda"
        )
        filtered_state = {
            k: v for k, v in state_dict.items()
            if not k.startswith("head.")
        }
        vit_model.load_state_dict(filtered_state, strict=False)
    elif args.img_type == "detr":
        detr_model = torch.hub.load('cooelf/detr', 'detr_resnet101_dc5', pretrained=True)
        detr_model.eval()

    all_images = [
        img for img in all_images
        if not img.startswith('.') and os.path.isdir(os.path.join(args.data_root, img))
    ]

    for idx, image in enumerate(tqdm(all_images)):
        path1 = os.path.join(args.data_root, image, "image.png")
        path2 = os.path.join(args.data_root, image, "choice_0.png")

        if os.path.isfile(path1):
            curr_dir = path1
        elif os.path.isfile(path2):
            curr_dir = path2
        else:
            print(f"⚠️ Warning: no image file found for {image}")
            continue
        feature = extract_features(args.img_type, curr_dir)
        tmp.append(feature.detach().cpu())
        name_map[str(image)] = idx
    
    # res = torch.cat(tmp).cpu()
    res = torch.cat(tmp, dim=0).cpu()
    print(res.shape)
    torch.save(res, os.path.join(args.output_dir, args.img_type +'.pth'))
    with open(os.path.join(args.output_dir, 'name_map.json'), 'w') as outfile:
        json.dump(name_map, outfile)