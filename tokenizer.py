from tokenizers import BertWordPieceTokenizer

def train_tokenizer(captions):
    print('Create training file...')
    train_tokenizer = [sample for samples in captions for sample in samples]
    with open('train_tokenizer.txt', 'a') as f:
        for sample in train_tokenizer:
            f.write(sample)
    # init
    bwpt = BertWordPieceTokenizer(vocab_file=None,
                                  unk_token='[UNK]',
                                  sep_token='[SEP]',
                                  cls_token='[CLS]',
                                  clean_text=True,
                                  handle_chinese_chars=True,
                                  strip_accents=True,
                                  lowercase=True,
                                  wordpieces_prefix='##')
    print('Tokenizer training...')
    bwpt.train(files=['train_tokenizer.txt'],
               vocab_size=30000,
               min_frequency=5,
               limit_alphabet=1000,
               special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]'])

    bwpt.save('.', 'captions')

    # initialization of a trained tokenizer
    tokenizer = BertWordPieceTokenizer('captions-vocab.txt')
    tokenizer.enable_truncation(max_length=16)
    print('Tokenizer is ready to use...')
    return tokenizer