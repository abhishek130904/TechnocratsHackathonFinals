# Dataset Guide: Where to Get Training Data

This guide explains where to find or create datasets for training your counterfeit detection model.

## 🎯 What You Need

- **Real Product Images**: Authentic, genuine products
- **Fake Product Images**: Counterfeit, replica products
- **Minimum**: 50-100 images per class (real/fake) per split (train/val/test)
- **Recommended**: 200+ images per class per split (1200+ total)

## 📚 Option 1: Public Datasets (If Available)

### Search for Existing Datasets

**Kaggle:**
- Search: "counterfeit detection", "fake product", "authenticity detection"
- URL: https://www.kaggle.com/datasets
- Look for: Product authenticity, brand verification datasets

**Google Dataset Search:**
- URL: https://datasetsearch.research.google.com/
- Search terms: "counterfeit products", "fake detection", "product authenticity"

**Academic Datasets:**
- Papers with Code: https://paperswithcode.com/
- Look for research papers on counterfeit detection
- Many papers include dataset links

**Note**: Public datasets for counterfeit detection are rare. You may need to create your own.

## 🛍️ Option 2: E-commerce Platforms (With Permission)

### Where to Find Images

**1. Official Brand Websites**
- ✅ Legitimate product images
- ✅ High quality
- ✅ Clearly authentic
- Example: Nike.com, Adidas.com, etc.

**2. E-commerce Marketplaces**
- Amazon (official sellers)
- eBay (verified sellers)
- Official brand stores on platforms

**3. Product Review Sites**
- Product images from reviews
- Often show real products in use

**⚠️ Important Legal Notes:**
- Always check terms of service
- Get permission if required
- Don't scrape without permission
- Respect copyright and usage rights

## 📸 Option 3: Create Your Own Dataset

### Step-by-Step Process

#### A. Collect Real Product Images

**Sources:**
1. **Your Own Products**
   - Take photos of products you own
   - Ensure they're authentic
   - Multiple angles, lighting conditions

2. **Friends/Family**
   - Ask to photograph their products
   - Document that they're authentic

3. **Retail Stores (With Permission)**
   - Ask store managers for permission
   - Photograph products in-store
   - Get written permission if possible

4. **Product Photography Services**
   - Hire photographers
   - Use stock photo services (with licensing)

#### B. Collect Fake Product Images

**Sources:**
1. **Replica Markets** (For Research Only)
   - Document replicas for research purposes
   - Clearly label as fake
   - Use for training only, not promotion

2. **Online Marketplaces**
   - Look for listings that explicitly state "replica" or "first copy"
   - Screenshot product images
   - Document source and label clearly

3. **Research Collaborations**
   - Partner with brands or researchers
   - They may have collections of known fakes

4. **Synthetic Generation**
   - Use image editing to create variations
   - Apply filters, distortions to real images
   - Create "fake-looking" versions

#### C. Organize Your Dataset

```
dataset/
├── train/
│   ├── real/
│   │   ├── nike_shoe_001.jpg
│   │   ├── nike_shoe_002.jpg
│   │   ├── adidas_shoe_001.jpg
│   │   └── ...
│   └── fake/
│       ├── fake_nike_001.jpg
│       ├── fake_nike_002.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Naming Convention:**
- Use descriptive names: `brand_product_type_number.jpg`
- Include metadata in filename if helpful
- Keep consistent naming

## 🤖 Option 4: Web Scraping (Advanced)

### Tools for Scraping

**Python Libraries:**
- `requests` + `BeautifulSoup` (for HTML)
- `selenium` (for JavaScript-heavy sites)
- `scrapy` (full scraping framework)

### Example Scraper Structure

```python
# Example: Scrape product images (with permission!)
import requests
from bs4 import BeautifulSoup
from pathlib import Path

def scrape_product_images(url, output_dir, label="real"):
    """
    Scrape product images from a website.
    
    ⚠️ WARNING: Only use with permission!
    Check robots.txt and terms of service.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find image tags
    images = soup.find_all('img', class_='product-image')
    
    output_path = Path(output_dir) / label
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        img_url = img['src']
        img_data = requests.get(img_url).content
        
        with open(output_path / f"product_{i:03d}.jpg", 'wb') as f:
            f.write(img_data)
```

**⚠️ Legal & Ethical Considerations:**
- Always check `robots.txt` (e.g., `website.com/robots.txt`)
- Read terms of service
- Respect rate limits
- Don't overload servers
- Get permission when possible
- Use for research/educational purposes only

## 🎨 Option 5: Data Augmentation (Expand Small Dataset)

If you have a small dataset, you can expand it:

### Image Augmentation

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

def augment_images(input_dir, output_dir, num_augmented=5):
    """
    Create augmented versions of images.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Load and augment each image
    for img_path in Path(input_dir).glob("*.jpg"):
        img = Image.open(img_path)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Generate augmented images
        for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
            if i >= num_augmented:
                break
            augmented_img = Image.fromarray(batch[0].astype('uint8'))
            augmented_img.save(
                output_dir / f"{img_path.stem}_aug_{i}.jpg"
            )
```

## 📊 Option 6: Synthetic Data Generation

### Using GANs or Image Generation

**Tools:**
- Stable Diffusion (for generating product images)
- DALL-E API (if available)
- Custom GANs trained on product images

**Approach:**
1. Generate "real-looking" product images
2. Generate "fake-looking" variations
3. Label appropriately

## 🎯 Recommended Approach for Beginners

### Start Simple:

1. **Week 1-2: Collect 50-100 Real Images**
   - Use your own products
   - Take photos with phone
   - Multiple products, multiple angles

2. **Week 2-3: Collect 50-100 Fake Images**
   - Document replicas (for research)
   - Screenshot online listings
   - Label clearly as fake

3. **Week 3: Organize Dataset**
   - Create folder structure
   - Rename files consistently
   - Split into train/val/test (80/10/10)

4. **Week 4: Train Model**
   - Start with small dataset
   - See if it works
   - Iterate and improve

### Quick Start Script

I'll create a script to help you organize your dataset:

```python
# dataset_organizer.py
from pathlib import Path
import shutil
import random

def organize_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Organize images into train/val/test splits.
    
    Args:
        source_dir: Directory with 'real' and 'fake' subdirectories
        output_dir: Where to create organized dataset
        train_ratio: Percentage for training (default 80%)
        val_ratio: Percentage for validation (default 10%)
        # test_ratio = 1 - train_ratio - val_ratio (default 10%)
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    for label in ['real', 'fake']:
        label_dir = source / label
        if not label_dir.exists():
            continue
        
        images = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
        random.shuffle(images)
        
        # Calculate splits
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Copy to organized structure
        for split, img_list in [('train', train_images), 
                                ('val', val_images), 
                                ('test', test_images)]:
            dest_dir = output / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img in img_list:
                shutil.copy(img, dest_dir / img.name)
        
        print(f"{label}: {n_train} train, {n_val} val, {len(test_images)} test")

# Usage:
# organize_dataset("raw_images", "dataset", train_ratio=0.8, val_ratio=0.1)
```

## 📝 Dataset Quality Tips

### Good Dataset Characteristics:

1. **Diversity**
   - Multiple product categories
   - Different lighting conditions
   - Various angles and backgrounds
   - Different image qualities

2. **Balance**
   - Similar number of real and fake images
   - Similar distribution across categories

3. **Quality**
   - Clear, in-focus images
   - Reasonable resolution (224x224 minimum)
   - Proper labeling (no mistakes!)

4. **Representativeness**
   - Images similar to what you'll see in production
   - Real-world conditions (not just studio photos)

## 🚀 Quick Start: Minimal Dataset

**If you just want to test the training pipeline:**

1. **Collect 20-30 images** (10-15 real, 10-15 fake)
2. **Organize into folders**
3. **Train with small dataset** (will overfit, but tests the pipeline)
4. **Expand dataset** as you get more images

## 📚 Resources

### Image Collection Tools:
- **Google Images**: Advanced search → Usage rights → "Labeled for reuse"
- **Unsplash/Pexels**: Free stock photos (for real products)
- **Flickr**: Creative Commons images

### Annotation Tools:
- **LabelImg**: For bounding boxes (if needed)
- **CVAT**: Computer Vision Annotation Tool
- **Roboflow**: Dataset management platform

### Dataset Management:
- **Roboflow**: https://roboflow.com/ (free tier available)
- **Label Studio**: https://labelstud.io/
- **DVC**: Data version control

## ⚠️ Important Reminders

1. **Legal Compliance**: Always respect copyright and terms of service
2. **Ethical Use**: Use datasets responsibly
3. **Privacy**: Don't include personal information in images
4. **Quality Over Quantity**: Better to have 100 good images than 1000 bad ones
5. **Documentation**: Keep notes on where images came from

## 🎯 Next Steps

1. **Choose your approach** (I recommend starting with your own photos)
2. **Collect initial dataset** (aim for 50-100 images per class)
3. **Organize into folder structure**
4. **Run training script**
5. **Iterate and improve**

---

**Remember**: You can start small! Even 50 images per class can train a basic model. You can always add more data later.

