import torch
from consts import EOS, PAD, UNK

def symbol_to_idx(sym_list: list) -> dict:
    symb_idx = {s: i for i, s in enumerate(sym_list)}
    return symb_idx

def idx_to_symbol(symb_idx: dict) -> list:
    sym_list = list({ k for k in symb_idx})
    return sym_list

def text_to_seq_per_char(text: str, symb_idx: dict) -> torch.IntTensor:
    seq = []
    for sym in text:
        idx = symb_idx.get(sym, None)
        if idx is not None:
            seq.append(idx)
        else:
            seq.append(symb_idx.get(UNK))
            print('not in vocab')
    seq.append(symb_idx.get(EOS))
    return torch.IntTensor(seq)

def seq_to_text(seq: torch.IntTensor, sym_list: list, remove_pad:bool = True):
    text = ""
    for idx in seq:
        symbol = sym_list[idx]
        # print(idx)
        # print(symbol)
        if remove_pad and symbol == PAD:
            symbol = ""
        text += symbol
    return text

if __name__ == "__main__":
    temp_ls = ['PAD', 'EOS', 'a', 'b', 'c', 'd', 'p', 'UNK'] #it is assumed that PAD is 0 and EOS is 1. UNK (unknown char) is last idx this is just for testing a sample vocab
    bruh = symbol_to_idx(temp_ls)
    print(text_to_seq_per_char('apple', bruh))

    temp_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    gaming = idx_to_symbol(temp_dict)
    test_tensor = torch.IntTensor([2, 6, 6, 6, 6, 1])
    print(seq_to_text(test_tensor, temp_ls))