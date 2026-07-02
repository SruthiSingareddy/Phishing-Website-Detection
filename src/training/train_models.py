"""
Advanced Model Training Pipeline
Trains URLNet, URL-Transformer, and ensemble models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from pathlib import Path
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our models
import sys
sys.path.append('..')
from models.urlnet import URLNet, URLNetLoss, create_urlnet_model
from models.url_transformer import URLTransformer, URLTokenizer
from preprocessing.feature_engineering import create_feature_pipeline


class URLDataset(Dataset):
    """Dataset class for URL data"""
    
    def __init__(self, urls: List[str], labels: List[int], tokenizer: URLTokenizer, max_length: int = 200):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        
        # Tokenize URL for transformer
        tokenized = self.tokenizer.tokenize_url(url)
        
        # Convert URL to character indices for URLNet
        char_indices = []
        for char in url[:self.max_length]:
            char_idx = ord(char) if ord(char) < 128 else 1  # UNK token
            char_indices.append(char_idx)
        
        # Pad or truncate
        if len(char_indices) < self.max_length:
            char_indices.extend([0] * (self.max_length - len(char_indices)))  # PAD token
        else:
            char_indices = char_indices[:self.max_length]
        
        return {
            'url': url,
            'char_indices': torch.tensor(char_indices, dtype=torch.long),
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'token_type_ids': tokenized['token_type_ids'],
            'label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(min(len(url), self.max_length), dtype=torch.long)
        }


class ModelTrainer:
    """Advanced model training pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        self.model_dir = Path(config['model_dir'])
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = URLTokenizer(max_length=config['max_length'])
        
        # Training history
        self.training_history = {
            'urlnet': {'train_loss': [], 'val_loss': [], 'val_accuracy': []},
            'transformer': {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        }
    
    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data"""
        print("Loading and preparing data...")
        
        # Load URLs
        train_df = pd.read_csv(f"{data_path}/train_urls.csv")
        test_df = pd.read_csv(f"{data_path}/test_urls.csv")
        
        # Split training data for validation
        train_urls, val_urls, train_labels, val_labels = train_test_split(
            train_df['url'].tolist(),
            train_df['label'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=train_df['label']
        )
        
        # Create datasets
        train_dataset = URLDataset(train_urls, train_labels, self.tokenizer, self.config['max_length'])
        val_dataset = URLDataset(val_urls, val_labels, self.tokenizer, self.config['max_length'])
        test_dataset = URLDataset(test_df['url'].tolist(), test_df['label'].tolist(), 
                                 self.tokenizer, self.config['max_length'])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                              shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], 
                               shuffle=False, num_workers=4)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_urlnet(self, train_loader: DataLoader, val_loader: DataLoader) -> URLNet:
        """Train URLNet model"""
        print("\n" + "="*50)
        print("Training URLNet Model")
        print("="*50)
        
        # Create model
        model = create_urlnet_model(self.config['urlnet'])
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = URLNetLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), 
                               lr=self.config['learning_rate'],
                               weight_decay=self.config['weight_decay'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=3)
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in pbar:
                char_indices = batch['char_indices'].to(self.device)
                labels = batch['label'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(char_indices, lengths)
                loss = criterion(outputs['logits'], labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs['probabilities'], 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100 * train_correct / train_total:.2f}%"
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    char_indices = batch['char_indices'].to(self.device)
                    labels = batch['label'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    
                    outputs = model(char_indices, lengths)
                    loss = criterion(outputs['logits'], labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs['probabilities'], 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Update history
            self.training_history['urlnet']['train_loss'].append(avg_train_loss)
            self.training_history['urlnet']['val_loss'].append(avg_val_loss)
            self.training_history['urlnet']['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Early stopping and model saving
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / 'urlnet_best.pth')
                print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(self.model_dir / 'urlnet_best.pth'))
        return model
    
    def train_transformer(self, train_loader: DataLoader, val_loader: DataLoader) -> URLTransformer:
        """Train URL Transformer model"""
        print("\n" + "="*50)
        print("Training URL Transformer Model")
        print("="*50)
        
        # Create model
        model = URLTransformer(
            vocab_size=self.tokenizer.vocab_size,
            **self.config['transformer']
        )
        model = model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), 
                               lr=self.config['learning_rate'],
                               weight_decay=self.config['weight_decay'])
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=3)
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs['logits'], labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs['probabilities'], 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100 * train_correct / train_total:.2f}%"
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(outputs['logits'], labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs['probabilities'], 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Update history
            self.training_history['transformer']['train_loss'].append(avg_train_loss)
            self.training_history['transformer']['val_loss'].append(avg_val_loss)
            self.training_history['transformer']['val_accuracy'].append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Early stopping and model saving
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / 'transformer_best.pth')
                print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(self.model_dir / 'transformer_best.pth'))
        return model
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, model_type: str) -> Dict[str, float]:
        """Evaluate model performance"""
        print(f"\nEvaluating {model_type} model...")
        
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                labels = batch['label'].to(self.device)
                
                if model_type == 'URLNet':
                    char_indices = batch['char_indices'].to(self.device)
                    lengths = batch['length'].to(self.device)
                    outputs = model(char_indices, lengths)
                else:  # Transformer
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    outputs = model(input_ids, attention_mask, token_type_ids)
                
                probabilities = outputs['probabilities'].cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities[:, 1])  # Probability of phishing
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
        auc = roc_auc_score(all_labels, all_probabilities)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc
        }
        
        print(f"{model_type} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.title(f'{model_type} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.model_dir / f'{model_type.lower()}_confusion_matrix.png', dpi=300)
        plt.close()
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # URLNet plots
        if self.training_history['urlnet']['train_loss']:
            axes[0, 0].plot(self.training_history['urlnet']['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.training_history['urlnet']['val_loss'], label='Val Loss')
            axes[0, 0].set_title('URLNet Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(self.training_history['urlnet']['val_accuracy'])
            axes[0, 1].set_title('URLNet Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].grid(True)
        
        # Transformer plots
        if self.training_history['transformer']['train_loss']:
            axes[1, 0].plot(self.training_history['transformer']['train_loss'], label='Train Loss')
            axes[1, 0].plot(self.training_history['transformer']['val_loss'], label='Val Loss')
            axes[1, 0].set_title('Transformer Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(self.training_history['transformer']['val_accuracy'])
            axes[1, 1].set_title('Transformer Validation Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy (%)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=300)
        plt.close()
    
    def save_training_config(self):
        """Save training configuration and results"""
        config_path = self.model_dir / 'training_config.json'
        
        save_config = {
            'config': self.config,
            'training_history': self.training_history,
            'device': str(self.device),
            'vocab_size': self.tokenizer.vocab_size
        }
        
        with open(config_path, 'w') as f:
            json.dump(save_config, f, indent=2)
        
        # Save tokenizer
        with open(self.model_dir / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def train_all_models(self, data_path: str):
        """Train all models in the pipeline"""
        print("Starting Advanced Phishing Detection Model Training")
        print("="*60)
        
        # Load data
        train_loader, val_loader, test_loader = self.load_data(data_path)
        
        # Train URLNet
        urlnet_model = self.train_urlnet(train_loader, val_loader)
        urlnet_metrics = self.evaluate_model(urlnet_model, test_loader, 'URLNet')
        
        # Train Transformer
        transformer_model = self.train_transformer(train_loader, val_loader)
        transformer_metrics = self.evaluate_model(transformer_model, test_loader, 'Transformer')
        
        # Plot training history
        self.plot_training_history()
        
        # Save configuration
        self.save_training_config()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"URLNet - Accuracy: {urlnet_metrics['accuracy']:.4f}, F1: {urlnet_metrics['f1_score']:.4f}")
        print(f"Transformer - Accuracy: {transformer_metrics['accuracy']:.4f}, F1: {transformer_metrics['f1_score']:.4f}")
        
        return {
            'urlnet': {'model': urlnet_model, 'metrics': urlnet_metrics},
            'transformer': {'model': transformer_model, 'metrics': transformer_metrics}
        }


def main():
    """Main training function"""
    # Training configuration
    config = {
        'model_dir': 'models',
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'epochs': 50,
        'patience': 7,
        'max_length': 200,
        
        # URLNet configuration
        'urlnet': {
            'vocab_size': 128,
            'embed_dim': 128,
            'num_filters': 256,
            'filter_sizes': [3, 4, 5, 6],
            'num_heads': 8,
            'hidden_dim': 512,
            'num_classes': 2,
            'max_length': 200,
            'dropout': 0.3
        },
        
        # Transformer configuration
        'transformer': {
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_hidden_layers': 6,
            'intermediate_size': 3072,
            'max_position_embeddings': 200,
            'num_classes': 2,
            'dropout': 0.1
        }
    }
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train models
    results = trainer.train_all_models('data/processed')
    
    return results


if __name__ == "__main__":
    results = main()