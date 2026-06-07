import sys
sys.path.append('/home/jupyter/datasphere/project/Untitled Folder')
import torch
from typing import Type
from torch import nn
from dataset import TextDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LanguageModel(nn.Module):
    def __init__(
        self, 
        dataset: TextDataset, 
        embed_size: int = 512, 
        hidden_size: int = 1024,
        rnn_type: Type = nn.RNN, 
        rnn_layers: int = 3,  
        dropout: float = 0.2,  
        bidirectional: bool = True
    ):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        :param dropout: dropout probability
        :param bidirectional: whether to use bidirectional RNN
        """
        super().__init__()
        self.dataset = dataset 
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout) # poprobuem dropout shtobi 4ek mb obu4itsa lutshe       
        self.rnn = rnn_type(embed_size, hidden_size, num_layers=rnn_layers, batch_first=True, dropout=dropout if rnn_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, self.vocab_size)



    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        indices = indices.to(self.embedding.weight.device)
        lengths = lengths.to(self.embedding.weight.device)
        embed = self.embedding(indices) # индекс токенов -> эмбеддинг
        embed = self.dropout(embed)
        true_embed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)  # эмбеды реальной длины
        true_embed_out, _ = self.rnn(true_embed) # пускаю данные через рекурентку, хиддены идут в ж*пу, они не нужны ща
        max_len = lengths.max().item()
        out, _ = pad_packed_sequence(true_embed_out,  batch_first=True, total_length=max_len) # перед финал выдчей распаковка данных до размерности 
        logits = self.linear(out) # вектор логитов размерности батч, длина послед и длина слварика

    
        # # This is a placeholder, you may remove it.
        # logits = torch.randn(
        #     indices.shape[0], indices.shape[1], self.vocab_size,
        #     device=indices.device
        # )
        """ 
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        return logits
    
    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = self.device
        # кодировка
        bos_pos = self.dataset.sp_model.bos_id() # индексы начала 
        if prefix:
            pref_pos = [bos_pos] + self.dataset.text2ids(prefix)  
        else:
            pref_pos = [bos_pos]
        resulted_pos_saved = pref_pos.copy() # спсиок копируем, там токены наши
        # кладем в модель
        bos_prefix_tensor = torch.tensor([pref_pos], dtype=torch.long, device=device)# prefix -> tensor
        prefix_embedding = self.embedding(bos_prefix_tensor) 
        _, hidden = self.rnn(prefix_embedding)    # v madelky
        #tut generiryu tokens
        i = len(pref_pos)
        while i < self.max_length:
            token_last = resulted_pos_saved[-1]
            if token_last == self.dataset.sp_model.eos_id(): # esli last token EOS to end
                break
            else:
                input_tensor = torch.tensor([[token_last]], dtype=torch.long, device=device)# token -> tensor 
                current_embedding = self.embedding(input_tensor) # tensor -> embed
                out, hidden = self.rnn(current_embedding, hidden) # madelka
                logits = self.linear(out.squeeze(1)) / temp # normalized exit 
                probabilities = torch.softmax(logits, dim=-1)   
                next_token = torch.multinomial(probabilities, num_samples=1).item() # sample each token from raspred
                resulted_pos_saved.append(next_token)
            i += 1
        # uberem EOS BOS
        if resulted_pos_saved[0] == bos_pos:
            resulted_pos_saved = resulted_pos_saved[1:]
        if resulted_pos_saved[::-1] == self.dataset.sp_model.eos_id():
            resulted_pos_saved = resulted_pos_saved[:-1]

        # Декодируем текст
        generated = self.dataset.ids2text(resulted_pos_saved)        

        # # This is a placeholder, you may remove it.
        # generated = prefix + ', а потом купил мужик шляпу, а она ему как раз.'
        # """
        # YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        # Encode the prefix (do not forget the BOS token!),
        # pass it through the model to accumulate RNN hidden state and
        # generate new tokens sequentially, sampling from categorical distribution,
        # until EOS token or reaching self.max_length.
        # Do not forget to divide predicted logits by temperature before sampling
        # """
        return generated