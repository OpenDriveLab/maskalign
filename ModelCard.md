# Model Card

This page lists the MaskAlign model weights. CLIP-L/14<sup>*</sup> denotes input 196 × 196 resolution image to CLIP-L/14. This will keep the same feature map size as the student model. PT epochs and FT Acc denotes pre-training epochs and fine-tuning accuracy on ImageNet-1K, respectively.


|  Model   | Teacher Model  | PT epochs | Link | FT Acc. |
|  :----:  | :----:         |  :----:   |:----:| :----:  |
| ViT-B/16 | CLIP-B/16 |  200 | [gdrive](https://drive.google.com/file/d/1hu_dlzOxVqS1Zx6W41aOu7qL2XqXOuqx/view?usp=share_link) | 85.4 |
| ViT-L/16 | CLIP-B/16 |  200 | [gdrive](https://drive.google.com/file/d/1hWdjhKso52K5M9xem0j81KJVhlg0oZov/view?usp=share_link) | 86.5 |
| ViT-L/16 | CLIP-L/14<sup>*</sup> | 200 | [gdrive](https://drive.google.com/file/d/1NdwxvQkaHk8axmThxDrQKTLJ8CF_QgM5/view?usp=share_link) | 87.4 |