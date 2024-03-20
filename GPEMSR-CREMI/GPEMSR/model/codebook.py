import torch
import torch.nn as nn
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args['num_codebook_vectors']
        self.latent_dim = args['latent_dim']
        self.beta = args['beta']

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()#[B,H,W,C]
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))#[B*H*W*C/d,1],[v],[B*H*W*C/d,v]---->[B*H*W*C/d,v]

        min_encoding_indices = torch.argmin(d, dim=1)#[B*H*W*C/d]
        z_q = self.embedding(min_encoding_indices).view(z.shape)#[B*H*W*C/d,d]--->[B,H,W,C]

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()#[B,C,H,W]

        return z_q, min_encoding_indices, loss

    def inference_lr(self, p):
        # p[B,H,W,1024]
        B,H,W,C = p.shape
        p = p.view(B*H*W,C).contiguous()
        soft_one_hot = F.softmax(p, dim=1)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=1)
        top_idx = top_idx.squeeze(1)#B*H*W
        z_q = self.embedding(top_idx).view(B,H,W,-1)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B,512,H,W]
        return z_q

if __name__ == '__main__':
    a = Codebook(args={
        'num_codebook_vectors': 1024,
        'latent_dim': 512,
        'beta': 1
    })
    b = torch.zeros(4,10,10,1024)
    print(b.shape)
    c = a.inference_lr(b)
    print(c.shape)
    b = torch.zeros(4, 512,10, 10)
    print(a(b)[0].shape)




