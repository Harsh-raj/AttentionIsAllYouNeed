from pathlib import Path

class Config:
  def get_config():
    return {
      "batch_size": 8,
      "num_epochs": 20,
      'lr': 10**-4,
      "seq_len": 350, #check seq_len required for this dataset
      "d_model": 512,
      "lang_src": "en",
      "lang_tgt": "hi",
      "moddel_folder": "weights",
      "model_filename": "tmodel_",
      "preload": None,
      "tokenizer_file": "tokenizer_{0}.json",
      "experiment_name": "runs/tmodel"      
    }
    
    def gt_weights_file_path(config, epoch: str):
      model_folder = config['model_folder']
      model_basename = config['model_basename']
      model_filename = f"{model_basename}{epoch}.pt"
      return str(Path('.') / model_folder / model_filename)