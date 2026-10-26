# Plant classifier: CNN stage

This project starts with a pretrained ResNet-18 CNN in PyTorch. It reads one folder per plant class, makes a reproducible per-class 70/15/15 train/validation/test split, uses training-only augmentation and class-weighted loss, and saves the best validation checkpoint plus test metrics.

The current source folders contain 1,706 JPEG photos:

| Class folder | Images |
|---|---:|
| `bougainvillea` | 192 |
| `copperleaf_plant` | 71 |
| `golden_net_bush` | 112 |
| `ixora_coccinea` | 308 |
| `lantana_camara` | 228 |
| `purple_false_erantheum` | 162 |
| `Tabernaemontana litoralis` | 319 |
| `tecomaria_capensis` | 93 |
| `Variegated Tecoma` | 132 |
| `wild jasmine` | 89 |

## Dataset layout

Keep the dataset outside GitHub. The directory passed to `--data-dir` should contain the 10 plant folders directly:

```text
images/
  bougainvillea/
  copperleaf_plant/
  golden_net_bush/
  ixora_coccinea/
  lantana_camara/
  purple_false_erantheum/
  Tabernaemontana litoralis/
  tecomaria_capensis/
  Variegated Tecoma/
  wild jasmine/
```

Folder names become the model's class labels. Spaces and capitalization are supported.

## Local run

Install a PyTorch build appropriate for your system from the [official PyTorch install selector](https://pytorch.org/get-started/locally/), then install the remaining requirements:

```bash
python -m pip install -r requirements.txt
python train_cnn.py --data-dir "C:/Playground/plant_classifier/images" --output-dir outputs/cnn
```

The script automatically selects CUDA when PyTorch detects an available NVIDIA GPU, and enables automatic mixed precision on CUDA. Otherwise, it uses the CPU.

## Kaggle or Colab

Upload the code to GitHub and the image dataset separately. In the hosted notebook, install the requirements (or use the platform's installed PyTorch), make the dataset visible to the runtime, then call `train_cnn.py` with that runtime's dataset path. Save the `outputs/cnn` directory to persistent storage before the session ends.

The training script uses ImageNet-pretrained ResNet-18 weights by default. The runtime needs internet access the first time it downloads those weights. To initialize randomly instead, pass `--no-pretrained`.

For the guided Google Colab flow, open `cnn_colab.ipynb`. It mounts Drive, checks the GPU, checks for all 10 class folders, copies the dataset to the runtime disk, runs the training script, and saves results back to Drive. Set the repository URL in the notebook after the GitHub repository exists.

## Outputs

- `best_resnet18.pt`: checkpoint selected by validation macro F1
- `test_report.json`: overall and per-class test metrics
- `test_confusion_matrix.csv`: test confusion matrix
- `training_history.json`: per-epoch metrics

The test split is evaluated once at the end. Use validation metrics when adjusting training settings.
