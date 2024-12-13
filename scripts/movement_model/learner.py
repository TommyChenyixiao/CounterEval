import torch
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):

    def __init__(self, graph, optim, scheduler, train_data, val_data, args, device, logger):
        
        # model
        self.graph = graph
        self.optim = optim
        self.scheduler = scheduler

        # gradient bound
        self.max_grad_clip = args.max_grad_clip
        self.max_grad_norm = args.max_grad_norm

        # data
        self.batch_size = args.batch_size
        self.train_data = train_data
        self.val_data = val_data
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.val_dataloader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.num_epochs = args.num_epochs
        self.args = args
        self.device = device
        self.logger = logger
        self.output_dir = args.output_dir

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'logs'))
    
    def train(self):
        self.logger.info("Start training")
        self.graph.train()

        num_batches = len(self.train_dataloader)
        for epoch in range(self.num_epochs):
            
            for i, (X, y) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                
                self.optim.zero_grad()

                recon_batch, mu, logvar = self.graph(y, X)
                loss, mse, kld = self.graph.loss(recon_batch, y, mu, logvar)
                loss.backward()
                
                # gradient clipping
                if self.max_grad_clip is not None:
                    torch.nn.utils.clip_grad_value_(self.graph.parameters(), self.max_grad_clip)
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.max_grad_norm)
                
                self.optim.step()
                self.writer.add_scalar('Loss/train', loss.item(), epoch * num_batches + i)
                self.writer.add_scalar('MSE/train', mse.item(), epoch * num_batches + i)
                self.writer.add_scalar('KLD/train', kld.item(), epoch * num_batches + i)
                if i % 50 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Batch {i+1}/{num_batches}, Loss: {loss.item()}")
            
            if self.scheduler is not None:
                self.scheduler.step()
            if epoch % 10 == 1:
                # checkpoint
                torch.save(self.graph.state_dict(), os.path.join(self.output_dir, f'checkpoint_{epoch}.pt'))
            
            self.validate(epoch)
                        
    
    def validate(self, epoch):
        self.logger.info("Start validation")
        self.graph.eval()
        
        num_batches = len(self.val_dataloader)
        with torch.no_grad():
            total_loss = 0
            total_mse = 0
            total_kld = 0
            for i, (X, y) in enumerate(self.val_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)

                recon_batch, mu, logvar = self.graph(y, X)
                loss, mse, kld = self.graph.loss(recon_batch, y, mu, logvar)
                total_loss += loss.item()
                total_mse += mse.item()
                total_kld += kld.item()
            
            self.writer.add_scalar('Loss/val', total_loss / len(self.val_dataloader), epoch)
            self.writer.add_scalar('MSE/val', total_mse / len(self.val_dataloader), epoch)
            self.writer.add_scalar('KLD/val', total_kld / len(self.val_dataloader), epoch)
            self.logger.info(f"Validation loss: {total_loss / len(self.val_dataloader)}")
        # self.writer.flush()