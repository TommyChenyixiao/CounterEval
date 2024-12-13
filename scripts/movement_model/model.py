import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, input_dim, num_heads, num_layers):

        super(Transformer, self).__init__()

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )

    def forward(self, x):
        return self.encoder(x)

class cVAE(nn.Module):

    def __init__(self, x_dim, context_dim, latent_dim):

        super(cVAE, self).__init__()

        self.x_dim = x_dim
        self.context_dim = context_dim

        # encoder
        self.fc1 = nn.Linear(x_dim + context_dim, 512)
        self.fc21 = nn.Linear(512, latent_dim)
        self.fc22 = nn.Linear(512, latent_dim)

        # decoder
        self.fc3 = nn.Linear(latent_dim + context_dim, 512)
        self.fc4 = nn.Linear(512, x_dim)

        self.elu = nn.ELU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x, context):
        
        input = torch.cat([x, context], dim=-1)
        h1 = self.elu(self.fc1(input))
        z_mu = self.fc21(h1)
        z_logvar = self.fc22(h1)
        
        return z_mu, z_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z, context):
        input = torch.cat([z, context], dim=-1)
        h3 = self.elu(self.fc3(input))
        return self.fc4(h3)
    
    def forward(self, x, context):
        mu, logvar = self.encode(x, context)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, context), mu, logvar

class MovementModel(nn.Module):

    def __init__(self, x_dim, context_dim, latent_dim, num_heads, num_layers):

        super(MovementModel, self).__init__()
        self.x_dim = x_dim
        self.laten_dim = latent_dim

        self.cvae = cVAE(x_dim, context_dim, latent_dim)
        self.transformer = Transformer(context_dim, num_heads, num_layers)
    
    def encode(self, x, context):
        context = self.transformer(context)
        return self.cvae.encode(x, context)

    def decode(self, z, context):
        context = self.transformer(context)
        return self.cvae.decode(z, context)

    def forward(self, x, context):
        return self.cvae(x, self.transformer(context))

    def loss(self, recon_x, x, mu, logvar):
        recon_x = recon_x.reshape(-1, self.x_dim)
        x = x.reshape(-1, self.x_dim)
        mu = mu.reshape(-1, self.laten_dim)
        logvar = logvar.reshape(-1, self.laten_dim)

        MSE = nn.MSELoss(reduction='mean')
        mse = MSE(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return mse + KLD, mse, KLD

if __name__ == "__main__":
    context = torch.randn(10, 23, 10)
    x = torch.randn(10, 23, 2)

    model = MovementModel(2, 10, 2, 2, 2)
    output, mu, logvar = model(x, context)

    print(model.loss(output, x, mu, logvar))