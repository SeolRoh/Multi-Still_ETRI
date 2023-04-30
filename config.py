dataset_config = {
    'train_size' : 0.8,
    'test_size' : 0.2,
    'shuffle' : True,
    'random_state' : 123,
}
    
audio_config = {
    'K': 1,
    'output_dim': 256,
    'use': 'hidden_state',
    'num_label': 7,
    'path': './TOTAL/',
    'cuda': 'cuda:0',
    # about 10s of wav files
    'max_length' : 512
}

text_config = {
    'K': 1,
    'output_dim': 256,
    'num_label': 7,
    'max_length': 128,
    'cuda': 'cuda:0',
    'freeze': False
}

multimodal_config = {
    'output_dim': 512,
    'num_labels': 7,
    'dropout': 0.1,
    'cuda': 'cuda:0',
    'use_threeway':False,
    'use_attention':False
}

train_config = {
    'epochs': 5,
    #'epochs': 30,
    'batch_size': 64,
    'lr': 5e-5,
    'accumulation_steps': 8,
    'cuda': 'cuda:0'
}

test_config = {
    'batch_size': 64,
    'cuda': 'cuda:0'
}


cross_attention_config = {
    'projection_dim': 768,
    'output_dim': 512,
    'num_labels': 7,
    'dropout': 0.1,
    'cuda': 'cuda:0',
    'num_heads': 8,
    #mini
    #'layers': 1,
    'layers': 3,
    'attn_dropout': 0,
    'relu_dropout': 0,
    'res_dropout': 0,
    'embed_dropout': 0
}


