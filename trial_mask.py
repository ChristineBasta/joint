import torch
import matplotlib.pyplot as plt
from fairseq import utils

batch_size = 5
F = 7
N = 5

#Create m matrix and mask some values
m = torch.randn(N, N)
m[0, 4] = 0
m[2, 1] = 0
m[4, 3] = 0


#Get zero indices
mask_idx = (m == 0).nonzero()

#Create x and mask same indices
x = torch.randn(batch_size, N, N, F)

x[:, mask_idx[:, 0], mask_idx[:, 1], :] = 0

#Print some results
print(x[0, :, :, 0])
#plt.imshow(x[0, :, :, 0])
#plt.show()

print(x[1, :, :, 0])
#plt.imshow(x[1, :, :, 0])
#plt.show()
print(x[2, :, :, 3])
#plt.imshow(x[2, :, :, 3])
#plt.show()

eyed=torch.eye(30,61)
#plt.imshow(eyed)
#plt.show()

rows=30
columns=61
mlen=30

tensor=torch.randn(rows, columns)
all_inf=utils.fill_with_neg_inf(tensor.new(rows, columns))
dec_attn_mask = (torch.triu(all_inf, 1 + mlen)
                 + torch.tril(all_inf, -3)).byte()[:, :]  # -1

#plt.imshow(dec_attn_mask)
#plt.show()
tensor2=torch.randn(columns, rows+columns)
rows=61
columns=30
all_inf=utils.fill_with_neg_inf(tensor.new(rows, columns))
dec_attn_mask1 = torch.triu(
    all_inf, diagonal=0)

#plt.imshow(dec_attn_mask1)
#plt.show()

dec_attn_mask = torch.tril(
    all_inf, diagonal=-3)
#plt.imshow(dec_attn_mask)
#plt.show()

#plt.imshow(dec_attn_mask1+dec_attn_mask)
#plt.show()





w_head_q=torch.randn(3,3)
print(w_head_q)
q,v,e=torch.chunk(w_head_q,3,1)
print(q)
print(v)
print(e)


w_head_q=torch.randn(5,5)
print(w_head_q)
qlen=4
w_head_q = w_head_q[-qlen:]
print(w_head_q)


print('prev_output_tokens')
prev_output_tokens=torch.randn(3,3,2)
print(prev_output_tokens)
prev_output_tokens = prev_output_tokens[:, -1:]
print(prev_output_tokens)

print('**********************************************************************************')
print('dec attn')
qlen=50
klen=100
mlen=50
word_emb=torch.randn(50,50)
emb=word_emb.new_ones(qlen, klen)
print(emb)
dec_attn_mask = word_emb.new_ones(qlen, klen).byte()[:, :,None]
print(word_emb.new_ones(qlen, klen))
print(dec_attn_mask)