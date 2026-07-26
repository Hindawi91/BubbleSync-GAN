import torch
import skimage
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
import skimage.morphology
import numpy as np
from skimage.filters import threshold_otsu
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def get_blobs_properties(images,labels,device,source_domain,target_domain, sigma=1.5, t=0.5, connectivity=1, percentage=0.20, min_blob_size = 200):

    ####### Select hyperparameter min_blob_size based on domain ########
  
    filtered_labeled_images = []

    blob_counts = []
    blob_mean_areas = []
    blob_std_areas = []

    for i in range(images.shape[0]):

        # Load the image
        image = images[i]
        label = labels[i]

        image = image.cpu()
        label = label.cpu()


        try:
            gray_image = skimage.color.rgb2gray(image)
        except:
            gray_image = image

        # Denoise the image with a Gaussian filter
        try:
            blurred_image = skimage.filters.gaussian(gray_image, sigma=sigma)
        except:
            blurred_image = skimage.filters.gaussian(gray_image.detach().cpu().numpy(), sigma=sigma)

        # Mask the image according to the threshold
        thresh = threshold_otsu(blurred_image)
        binary_mask = blurred_image < thresh

        # Perform connected component analysis
        # labeled_image, count = skimage.measure.label(binary_mask, connectivity=connectivity, return_num=True)

        ########### Filter out objects with area smaller than p*mean(areas) ###########

        # object_features = skimage.measure.regionprops(labeled_image)
        # object_areas = [objf["area"] for objf in object_features]
        # mean_area = np.mean(object_areas)
        # min_area = percentage * mean_area

        ###############################################################################

        ########### Filter out objects with area smaller than min_area ################
        #   negative = 0 | positive = 1 ---- e.g if exp is DS2_to_DS3 DS2 is negative with label = 0

        if label == 0:
            if source_domain == 'DS1':
                min_area = 1000
            elif source_domain == 'DS2':
                min_area = 150
            elif source_domain == 'DS3':
                min_area = 300
        elif label == 1:
            if target_domain == 'DS1':
                min_area = 1000
            elif target_domain == 'DS2':
                min_area = 150
            elif target_domain == 'DS3':
                min_area = 300

             
        # min_area = min_blob_size

        ###############################################################################

        object_mask = skimage.morphology.remove_small_objects(binary_mask, min_size=min_area)
        filtered_labeled_image, filtered_count = skimage.measure.label(object_mask, connectivity=connectivity, return_num=True)

        object_features = skimage.measure.regionprops(filtered_labeled_image)
        object_areas = [objf["area"] for objf in object_features]

        # Calculate filtered mean and std areas
        # filtered_mean_areas = np.mean(object_areas)
        # filtered_std_areas = np.std(object_areas)


        object_areas = torch.tensor(object_areas, dtype=torch.float32)

        filtered_mean_areas = torch.mean(object_areas) if not torch.isnan(torch.mean(object_areas)) else 0.0
        filtered_std_areas = torch.std(object_areas) if not torch.isnan(torch.std(object_areas)) else 0.0


        blob_counts.append(filtered_count)
        blob_mean_areas.append(filtered_mean_areas)
        blob_std_areas.append(filtered_std_areas)

        # filtered_labeled_images.append(filtered_labeled_image)

    # Convert the results back to PyTorch tensors

    blob_counts = torch.tensor(blob_counts, dtype=torch.float32).to(device)
    blob_mean_areas = torch.tensor(blob_mean_areas, dtype=torch.float32).to(device)
    blob_std_areas = torch.tensor(blob_std_areas, dtype=torch.float32).to(device)

    # filtered_labeled_images = torch.tensor(filtered_labeled_images, dtype=torch.float32)
    # torchvision.utils.save_image(filtered_labeled_images, f'filtered_labeled_images.png', nrow=1, padding=2, normalize=True)

    return blob_counts,blob_mean_areas,blob_std_areas

def load_images_as_batches(data_dir, batch_size=8, shuffle=True):


    # Define image transformations (you can customize these as needed)
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),               # Convert images to PyTorch tensors
    ])

    # Load images from the directory
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader




# Code for debugging:

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = load_images_as_batches(data_dir='/home/local/ASUAD/falhinda/NEW_FPGAN_Exp/FPGAN_Training/new_DS2_2_DS3/data/brats/syn/train', 
        batch_size=8, shuffle=True)    

    for images, labels in dataloader:
        print(images.shape)
        print(labels.shape)

        blob_counts,blob_mean_areas,blob_std_areas = get_blobs_properties(images=images,labels=labels,device = device,source_domain = "DS2",target_domain= "DS3")
        # print(results)

        print(f"{blob_counts.shape=}")
        print(f"{blob_mean_areas.shape=}")
        print(f"{blob_std_areas.shape=}")

        print(f"{blob_counts=}")
        print(f"{blob_mean_areas=}")
        print(f"{blob_std_areas=}")

        print(labels)
        torchvision.utils.save_image(images, f'original_imgs.png', nrow=1, padding=2, normalize=True)

        break


