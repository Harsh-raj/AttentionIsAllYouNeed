import torch.nn as nn
import torch.utils.tensorboard import SummaryWriter

class TrainingLoop:
  def get_model(config, vocab_src_len, vocab_tgt_len):
    model = BuildTransformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

  def train_model(config):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    Pth(config['model_folder']).mkdir(parent=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size().to(device))
    
    # tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr', eps=1e-9])
    
    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
      model_filename = get_weights_file_path(config, config['preload'])
      print(f'Preloading model {model_filename}')
      state = torch.load(model_filename)
      initial_epoch = state['epoch'] + 1
      optimizer.load_state_dict(state['optimizer_state_dict'])
      global_step = state['global_step']
      
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_soothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epoch']):
      batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
      for batch in batch_iterator:
        model.train()
        
        encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
        decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
        encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
        decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
        
        # Run the tensor through the tranformer
        encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #(B, seq_len, d_model)
        proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)
        
        label = batch['label'].to(device) #(B, seq_len)
        
        # (B, seq_len, tgt_vocab_size) --> (B * seq_len, tgt_vocab_size)
        loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()))
        batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
        
        # Log the loss
        writer.add_scalar('train loss', loss.item(), global_step)
        writer.flush()
        
        #Backpropagate the loss
        loss.backward()
        
        # Upgrade the weights
        optimizer.step()
        optimizer.zero_grad()
        
        global_step += 1
        
      run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
      
      # Save the model at the end of every epoch
      model_filename = get_weights_file_path(config, f'{epoch:02d}')
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }, model_filename)
      
if __name__ == '__main__':
  config = get_config()
  train_model(config)