import cv2
import torch
import numpy as np
import rasterio
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from guided_diffusion.guided_diffusion.image_datasets import _list_image_files_recursively
import pdb


def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type == 'ddpm':
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'deeplab']:
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform
                 

class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self, 
        X_spatial: torch.Tensor, 
        y_data: torch.Tensor,
        X_spectral:torch.Tensor
    ): 
        self.X_spatial = X_spatial
        self.y_data = y_data
        self.X_spectral =X_spectral

    def __getitem__(self, index):
      return self.X_spatial[index], self.y_data[index],self.X_spectral[index]

    def __len__(self):
        return len(self.X_spatial)


class ImageLabelDataset(Dataset):
   # modify on JUne 21, seperating data_dir and label_dir, rgb and hyperspectral cubes in the same folder dir
   # while labels in another dir
    #modify on April 30, add train_data pathches
    ''' 
    :param data_dir: path to a folder with images and their annotations. 
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
        transform=None,
        image_sort = True

    ):
        super().__init__()

        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)

        
        if (image_sort):
          self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]
      
        """
        # label will be in a different data dir 
        self.label_paths = [
          os.path.join(label_dir, os.path.basename(image_path).split('.')[0] + '.npy')
            for image_path in self.image_paths]

        """
        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths]

        
        #data_cube is dimension (64,64,144)
        self.data_cube_path = [
            '.'.join(image_path.split('.')[:-1] + ['npz'])
            for image_path in self.image_paths
        ]
        
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
       

        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert pil_image.size[0] == pil_image.size[1], \
               f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        
        data_cube_path = self.data_cube_path[idx]
        with np.load(data_cube_path) as data:
            data_cube = data['arr1']
        
        tensor_label = torch.from_numpy(label)
        tensor_data_cube = torch.from_numpy(data_cube)
        return tensor_image, tensor_label,tensor_data_cube
        #return tensor_image, tensor_label



class HyperspectralDataset(Dataset):
    """
    Dataset class for hyperspectral image processing.
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images=-1,
        transform=None,
        image_sort=True,
        channel_groups=None

    ):
        super().__init__()

        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)

        if image_sort:
            self.image_paths = sorted(self.image_paths)
 
        if num_images > 0:
            print(f"Taking first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        # Corresponding label paths
        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

        # Corresponding hyperspectral cube paths
        self.data_cube_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npz'])
            for image_path in self.image_paths
        ]

        # Define channel groups for hyperspectral data
        self.channel_groups = channel_groups
     
  
    @staticmethod
    def stretch_minmax(band, lower_percent=2, upper_percent=98):
        """Apply contrast stretching based on given percentiles."""
        p_low, p_high = np.percentile(band, (lower_percent, upper_percent))
        return np.clip((band - p_low) / (p_high - p_low + 1e-10), 0, 1)  # Avoid division by zero

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        pil_RGB = Image.open(image_path)
        pil_RGB = pil_RGB.convert("RGB")
        assert pil_RGB.size[0] == pil_RGB.size[1], \
               f"Only square images are supported: ({pil_RGB.size[0]}, {pil_RGB.size[1]})"

        tensor_RGB = self.transform(pil_RGB)

        # Load the corresponding label
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        tensor_label = torch.from_numpy(label)

        # Load hyperspectral data cube
        data_cube_path = self.data_cube_paths[idx]
        with np.load(data_cube_path) as data:
            data_cube = data['arr1']  # shape: (channels, resolution, resolution)

        # Generate RGB-like images from hyperspectral data using channel groups
        grouped_images = []
        for group in self.channel_groups:
            selected_channels = data_cube[group,:, :]  # Extract selected channels
            normalized_channels = np.stack(
                [self.stretch_minmax(selected_channels[i,:,:]) for i in reversed(range(3))],
                axis=-1
            )  
            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray((normalized_channels * 255).astype('uint8'))
            if self.transform:
                transformed_image = self.transform(pil_image)
            else:
                transformed_image = transforms.ToTensor()(pil_image)  
            grouped_images.append(transformed_image)  # Store transformed image 


        return tensor_RGB, grouped_images, tensor_label


class ImageLabelSegmunich(Dataset):
    '''
   # modify on Jan 20,2015  for segmunich dataset 
    :param data_dir: path to a folder with images and their annotations. 
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        image_dir: str,
        label_dir:str,
        csv_file:str,
        resolution: int,
        num_images= -1,
        transform=None,

    ):
        super().__init__()
            

        self.resolution = resolution
        self.transform = transform

        # Read the CSV file to get image file names
        df = pd.read_csv(csv_file)
        self.image_paths = [os.path.join(image_dir, fname) for fname in df['filenames']]
        self.label_paths = [os.path.join(label_dir, fname) for fname in df['filenames']]
        """
        # Predefined mapping
        self.mapping = {
            0: 0,
            21: 1,
            22: 2,
            23: 3,
            31: 4,
            5: 5,
            32: 6,
            33: 7,
            41: 8,
            13: 9,
            14: 10,
            11: 11,
            12: 12
        }
        """
        # Predefined mapping if using ignore index = 255
        self.mapping = {
            0: 255,
            21: 0,
            22: 1,
            23: 2,
            31: 3,
            5: 4,
            32: 5,
            33: 6,
            41: 7,
            13: 8,
            14: 9,
            11: 10,
            12: 11
        }
  
        if num_images > 0:
            print(f"Take  {len(self.image_paths)} images...")


    def __len__(self):
        return len(self.image_paths)



    @staticmethod
    def stretch_minmax(band, lower_percent=2, upper_percent=98):
        p_low, p_high = np.percentile(band, (lower_percent, upper_percent))
        return np.clip((band - p_low) / (p_high - p_low + 1e-10), 0, 1)

    @staticmethod
    def read_image(image_path):
        """Static method to read and process an image."""
        with rasterio.open(image_path) as image:
            # Read all channels into a NumPy array
            img = image.read()

            # Extract RGB channels (assuming specific indices for R, G, B)
            r_channel = img[2, :, :]  # Adjust index as needed for your dataset
            g_channel = img[1, :, :]
            b_channel = img[0, :, :]

            r_channel = ImageLabelSegmunich.stretch_minmax(r_channel)
            g_channel = ImageLabelSegmunich.stretch_minmax(g_channel)
            b_channel = ImageLabelSegmunich.stretch_minmax(b_channel)
            # Stack channels into an RGB image
            rgb_image = np.stack((r_channel, g_channel, b_channel), axis=-1)
            #epsilon = 1e-10
            # Normalize the image to [0, 1]
            #rgb_image = rgb_image / (np.max(rgb_image) +epsilon)

            # Convert to PIL Image
            return Image.fromarray((rgb_image * 255).astype('uint8'))

    @staticmethod
    def read_label(label_path):
        """Static method to read and process a label."""
        with rasterio.open(label_path) as mask:
            # Read the label data
            label = mask.read()
            label = label[0, :, :]  # Use the first channel if applicable
            return torch.from_numpy(label)

    @staticmethod
    def remap_target(target, mapping):
        """Static method to remap target values using the mapping."""
        remapped = target.clone()  # Clone to avoid modifying the original tensor
        for original_value, new_value in mapping.items():
            remapped[target == original_value] = new_value
        return remapped

    def __getitem__(self, idx):
        #pdb.set_trace()
        # Load and process the image
        image_path = self.image_paths[idx]
        pil_image = self.read_image(image_path)

        # Save the original PIL image before applying transformations 
        """
        save_dir = "datasets/segmunich_true/train"
        save_path = os.path.join(save_dir, f"image_{idx}.png")
        pil_image.save(save_path)
        """
        # Apply transformations to the image
        tensor_image = self.transform(pil_image)

        # Load and process the label
        label_path = self.label_paths[idx]
        tensor_label = self.read_label(label_path)

        # Remap the target using the mapping
        tensor_label_remapped = self.remap_target(tensor_label, self.mapping)

        return tensor_image, tensor_label_remapped


class ImageLabelSegmunichGrouped(Dataset):
    def __init__(self, image_dir, label_dir, csv_file, resolution, num_images=-1, transform=None):
        self.dataset = ImageLabelSegmunich(image_dir, label_dir, csv_file, resolution, num_images, transform)
    
    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def group_channels(img):
        return [np.stack([ImageLabelSegmunich.stretch_minmax(img[i]) for i in group], axis=-1) for group in [(0,1,2), (3,4,5)]]

    def __getitem__(self, idx):
        img = ImageLabelSegmunich.read_image(self.dataset.image_paths[idx])
        groups = tuple(Image.fromarray((g * 255).astype('uint8')) for g in self.group_channels(img))
        tensor_images = tuple(self.dataset.transform(img) if self.dataset.transform else img for img in groups)
        label = self.dataset.read_label(self.dataset.label_paths[idx])
        return tensor_images, label


class OneraDataset(Dataset):
    def __init__(self, csv_file,basic_dir, resolution, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with file paths.
            transform (callable, optional): Optional transforms to be applied on the images and labels.
        """
        self.data = pd.read_csv(csv_file)
        self.resolution = resolution
        self.transform = transform
        self.basic_dir = basic_dir
        self.mapping = {
            1:0,
            2:1
        }

    def __len__(self):
        return len(self.data)



    @staticmethod
    def read_img_npy(patch_path):
        # Load .npy files
        patch = np.load(patch_path)  # Assuming HxWxC format
  
        # Extract individual bands
        patch_r = patch[:, :, 11]  # Third channel
        patch_g = patch[:, :, 7]  # Second channel
        patch_b = patch[:, :, 3]  # First channel


        patch_r = ImageLabelSegmunich.stretch_minmax(patch_r)
        patch_g = ImageLabelSegmunich.stretch_minmax(patch_g) 
        patch_b = ImageLabelSegmunich.stretch_minmax(patch_b)
 
      
        patch_image = np.stack((patch_r, patch_g, patch_b), axis=-1)

        return Image.fromarray((patch_image * 255).astype('uint8'))
        

    @staticmethod
    def read_label_npy(label_path):
        label = np.load(label_path)  # Assuming HxWxC format
  
        return torch.from_numpy(label)
        
  

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        # Get file paths from the row
        patch_1_path = self.data.iloc[idx]['img1_path']
        patch_2_path = self.data.iloc[idx]['img2_path']
        label_path = self.data.iloc[idx]['label_path']

        patch1_img = OneraDataset.read_img_npy(os.path.join(self.basic_dir,patch_1_path))
        patch2_img = OneraDataset.read_img_npy(os.path.join(self.basic_dir,patch_2_path))
        tensor_patch1 = self.transform(patch1_img)
        tensor_patch2 = self.transform(patch2_img)
        tensor_label = OneraDataset.read_label_npy(os.path.join(self.basic_dir,label_path))  
        # Remap the target using the mapping
        tensor_label_remapped = ImageLabelSegmunich.remap_target(tensor_label, self.mapping)

        return tensor_patch1, tensor_patch2, tensor_label_remapped


class InMemoryImageLabelDataset(Dataset):
    ''' 

    Same as ImageLabelDataset but images and labels are already loaded into RAM.
    It handles DDPM/GAN-produced datasets and is used to train DeepLabV3. 

    :param images: np.array of image samples [num_images, H, W, 3].
    :param labels: np.array of correspoding masks [num_images, H, W].
    :param resolution: image and mask output resolusion.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''

    def __init__(
            self,
            images: np.ndarray, 
            labels: np.ndarray,
            resolution=256,
            transform=None
    ):
        super().__init__()
        assert  len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        assert image.size[0] == image.size[1], \
               f"Only square images are supported: ({image.size[0]}, {image.size[1]})"

        tensor_image = self.transform(image)
        label = self.labels[idx]
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label



class BerlinFusionDataset(Dataset):

    """
    Dataset class for RGB + PCA + SAR modalities and label (.npy).
    Assumes filenames like rgb0.png, pca0.png, sar0.png, 0.npy
    """

    def __init__(self, data_dir, resolution, num_images=-1, transform=None):
        super().__init__()

        self.resolution = resolution
        self.transform = transform
        self.data_dir = data_dir

        # Collect all valid indices where all 4 files exist
        self.indices = []
        for filename in os.listdir(data_dir):

            if filename.startswith("rgb") and filename.endswith(".png"):
                idx = filename[3:-4]  # remove "rgb" and ".png"
                rgb_path = os.path.join(data_dir, f"rgb{idx}.png")
                pca_path = os.path.join(data_dir, f"pca{idx}.png")
                sar_path = os.path.join(data_dir, f"sar{idx}.png")
                label_path = os.path.join(data_dir, f"{idx}.npy")
                if os.path.exists(rgb_path) and os.path.exists(pca_path) and \
                   os.path.exists(sar_path) and os.path.exists(label_path):
                    self.indices.append(idx)

        self.indices = sorted(self.indices, key=lambda x: int(x))  # sort numerically
        
        if num_images > 0:
            print(f"Taking first {num_images} samples...")
            self.indices = self.indices[:num_images]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        #pdb.set_trace()
        index = self.indices[idx]

        # Build paths
        rgb_path = os.path.join(self.data_dir, f"rgb{index}.png")
        pca_path = os.path.join(self.data_dir, f"pca{index}.png")
        sar_path = os.path.join(self.data_dir, f"sar{index}.png")
        label_path = os.path.join(self.data_dir, f"{index}.npy")

        # Load images
        rgb = Image.open(rgb_path).convert("RGB")
        pca = Image.open(pca_path).convert("RGB")
        sar = Image.open(sar_path).convert("RGB")

        if self.transform:
            rgb = self.transform(rgb)
            pca = self.transform(pca)
            sar = self.transform(sar)

        # Load label and resize
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        tensor_label = torch.from_numpy(label)

        return {
            "rgb": rgb,
            "pca": pca,
            "sar": sar,
            "label": tensor_label,
            "index": index
        }


class AugsburgFusionDataset(Dataset):
    """
    Dataset class for RGB + PCA + SAR + DSM modalities and label (.npy).
    Assumes filenames like rgb0.png, pca0.png, sar0.png, dsm0.png, 0.npy
    """

    def __init__(self, data_dir, resolution, num_images=-1, transform=None):
        super().__init__()

        self.resolution = resolution
        self.transform = transform
        self.data_dir = data_dir

        # Collect all valid indices where all 5 files exist
        self.indices = []
        for filename in os.listdir(data_dir):
            if filename.startswith("rgb") and filename.endswith(".png"):
                idx = filename[3:-4]  # remove "rgb" and ".png"

                rgb_path = os.path.join(data_dir, f"rgb{idx}.png")
                pca_path = os.path.join(data_dir, f"pca{idx}.png")
                sar_path = os.path.join(data_dir, f"sar{idx}.png")
                dsm_path = os.path.join(data_dir, f"dsm{idx}.png")
                label_path = os.path.join(data_dir, f"{idx}.npy")

                if all(os.path.exists(p) for p in [rgb_path, pca_path, sar_path, dsm_path, label_path]):
                    self.indices.append(idx)

        self.indices = sorted(self.indices, key=lambda x: int(x))  # sort numerically

        if num_images > 0:
            print(f"Taking first {num_images} samples...")
            self.indices = self.indices[:num_images]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]

        # Build paths
        rgb_path = os.path.join(self.data_dir, f"rgb{index}.png")
        pca_path = os.path.join(self.data_dir, f"pca{index}.png")
        sar_path = os.path.join(self.data_dir, f"sar{index}.png")
        dsm_path = os.path.join(self.data_dir, f"dsm{index}.png")
        label_path = os.path.join(self.data_dir, f"{index}.npy")

        # Load images
        rgb = Image.open(rgb_path).convert("RGB")
        pca = Image.open(pca_path).convert("RGB")
        sar = Image.open(sar_path).convert("RGB")
        dsm = Image.open(dsm_path).convert("RGB")

        if self.transform:
            rgb = self.transform(rgb)
            pca = self.transform(pca)
            sar = self.transform(sar)
            dsm = self.transform(dsm)

        # Load label and resize
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        tensor_label = torch.from_numpy(label)

        return {
            "rgb": rgb,
            "pca": pca,
            "sar": sar,
            "dsm": dsm,
            "label": tensor_label,
            "index": index
        }
