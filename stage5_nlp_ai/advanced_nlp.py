"""
Stage 5: Advanced NLP and AI
Author: Jajitha
Course: Creating Your First Python Program - UST
Description: Transformers, LLMs and HCI in NLP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import re
from collections import Counter


# =============================================
# 1. TEXT PROCESSING PIPELINE
# =============================================

class TextProcessor:
    """
    Advanced text processing pipeline
    simulating transformer preprocessing.
    
    Author: Jajitha
    """
    
    def __init__(self):
        self.vocab = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        self.vocab.update(self.special_tokens)
        self.next_id = len(self.special_tokens)
    
    def tokenize(self, text):
        """Basic wordpiece-style tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s가-힣]', '', text)
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        
        for token, count in token_counts.most_common():
            if token not in self.vocab and count >= 2:
                self.vocab[token] = self.next_id
                self.next_id += 1
        
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def encode(self, text, max_length=128):
        """
        Encode text to token IDs
        simulating BERT encoding
        """
        tokens = ['[CLS]'] + self.tokenize(text) + ['[SEP]']
        
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        
        # Padding or truncation
        if len(ids) < max_length:
            attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            ids = ids + [self.vocab['[PAD]']] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
            attention_mask = [1] * max_length
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask,
            'token_count': sum(attention_mask)
        }
    
    def decode(self, ids):
        """Decode token IDs back to text"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []
        for id in ids:
            token = id_to_token.get(id, '[UNK]')
            if token not in ['[PAD]', '[CLS]', '[SEP]']:
                tokens.append(token)
        return ' '.join(tokens)


# =============================================
# 2. ATTENTION MECHANISM
# =============================================

class SelfAttention:
    """
    Self-attention mechanism implementation.
    Core component of Transformer architecture.
    
    Author: Jajitha
    """
    
    def __init__(self, d_model=64, num_heads=8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        np.random.seed(42)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def softmax(self, x):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention
        Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
        """
        scores = np.matmul(Q, K.transpose(-2, -1))
        scores = scores / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + (mask * -1e9)
        
        attention_weights = self.softmax(scores)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, x, mask=None):
        """Forward pass through self-attention"""
        seq_len = x.shape[0]
        
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        output, weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        output = np.matmul(output, self.W_o)
        
        return output, weights
    
    def visualize_attention(self, weights, tokens):
        """Visualize attention weights as heatmap"""
        plt.figure(figsize=(10, 8))
        plt.imshow(weights, cmap='Blues', aspect='auto')
        plt.colorbar(label='Attention Weight')
        
        if tokens:
            plt.xticks(
                range(len(tokens)),
                tokens,
                rotation=45,
                ha='right'
            )
            plt.yticks(range(len(tokens)), tokens)
        
        plt.title(
            'Self-Attention Weights Visualization\nAuthor: Jajitha',
            fontweight='bold'
        )
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.tight_layout()
        plt.savefig('results/attention_weights.png', dpi=150)
        plt.show()
        print("Attention visualization saved!")


# =============================================
# 3. LLM INTERACTION SIMULATOR
# =============================================

class LLMSimulator:
    """
    Simulates LLM interaction patterns
    for research and HCI studies.
    
    Author: Jajitha
    """
    
    def __init__(self, model_name="GPT-Simulator"):
        self.model_name = model_name
        self.conversation_history = []
        self.response_times = []
        self.token_counts = []
    
    def generate_response(self, prompt, max_tokens=100):
        """Simulate LLM response generation"""
        start_time = datetime.now()
        
        # Simulate processing
        tokens_generated = min(
            len(prompt.split()) * 2,
            max_tokens
        )
        
        # Template responses based on topic
        responses = {
            'korean': "한국어 학습에 관한 질문이군요! Korean language learning requires consistent practice with TOPIK preparation materials.",
            'gks': "GKS scholarship applications require strong academic records, research experience, and Korean language proficiency.",
            'nlp': "Natural Language Processing combines linguistics and machine learning to enable computers to understand human language.",
            'research': "Research papers in NLP typically follow IMRaD structure: Introduction, Methods, Results, and Discussion.",
            'default': f"This is a simulated response from {self.model_name} for research purposes in HCI studies."
        }
        
        prompt_lower = prompt.lower()
        if 'korean' in prompt_lower or 'topik' in prompt_lower:
            response = responses['korean']
        elif 'gks' in prompt_lower or 'scholarship' in prompt_lower:
            response = responses['gks']
        elif 'nlp' in prompt_lower or 'language' in prompt_lower:
            response = responses['nlp']
        elif 'research' in prompt_lower or 'paper' in prompt_lower:
            response = responses['research']
        else:
            response = responses['default']
        
        end_time = datetime.now()
        response_time = (end_time - start_time).microseconds / 1000
        
        self.conversation_history.append({
            'role': 'user',
            'content': prompt,
            'timestamp': start_time.isoformat()
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': end_time.isoformat(),
            'tokens': tokens_generated,
            'response_time_ms': response_time
        })
        
        self.response_times.append(response_time)
        self.token_counts.append(tokens_generated)
        
        return response, tokens_generated, response_time
    
    def get_conversation_stats(self):
        """Get statistics about conversation"""
        user_messages = [
            m for m in self.conversation_history
            if m['role'] == 'user'
        ]
        assistant_messages = [
            m for m in self.conversation_history
            if m['role'] == 'assistant'
        ]
        
        return {
            'total_turns': len(user_messages),
            'total_tokens': sum(self.token_counts),
            'avg_response_time_ms': np.mean(self.response_times),
            'avg_tokens_per_response': np.mean(self.token_counts),
            'conversation_length': len(self.conversation_history)
        }
    
    def export_conversation(self, filepath='results/conversation.json'):
        """Export conversation history to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                self.conversation_history,
                f,
                indent=4,
                ensure_ascii=False
            )
        print(f"Conversation exported to {filepath}")


# =============================================
# 4. HCI STUDY ANALYZER
# =============================================

class HCIStudyAnalyzer:
    """
    Analyzes Human-Computer Interaction patterns
    in LLM usage for research purposes.
    
    Author: Jajitha
    """
    
    def __init__(self):
        self.interaction_data = []
        self.user_satisfaction = []
    
    def record_interaction(
        self,
        user_id,
        prompt_length,
        response_time,
        satisfaction_score,
        task_completed
    ):
        """Record a user interaction"""
        self.interaction_data.append({
            'user_id': user_id,
            'prompt_length': prompt_length,
            'response_time': response_time,
            'satisfaction': satisfaction_score,
            'task_completed': task_completed,
            'timestamp': datetime.now().isoformat()
        })
        self.user_satisfaction.append(satisfaction_score)
    
    def generate_synthetic_study_data(self, n_users=50):
        """Generate synthetic HCI study data"""
        np.random.seed(42)
        
        for i in range(n_users):
            n_interactions = np.random.randint(5, 20)
            for _ in range(n_interactions):
                prompt_length = np.random.randint(10, 200)
                response_time = np.random.uniform(0.5, 5.0)
                
                satisfaction = min(10, max(1,
                    7 - response_time * 0.5 +
                    np.random.normal(0, 1)
                ))
                
                task_completed = np.random.choice(
                    [True, False],
                    p=[0.75, 0.25]
                )
                
                self.record_interaction(
                    f'U{i:03d}',
                    prompt_length,
                    response_time,
                    satisfaction,
                    task_completed
                )
        
        print(f"Generated {len(self.interaction_data)} interactions from {n_users} users")
    
    def analyze_patterns(self):
        """Analyze HCI patterns"""
        df = pd.DataFrame(self.interaction_data)
        
        print("\n" + "="*50)
        print("HCI STUDY ANALYSIS RESULTS")
        print("="*50)
        print(f"Total interactions: {len(df)}")
        print(f"Unique users: {df['user_id'].nunique()}")
        print(f"Avg satisfaction: {df['satisfaction'].mean():.2f}/10")
        print(f"Task completion rate: {df['task_completed'].mean():.2%}")
        print(f"Avg response time: {df['response_time'].mean():.2f}s")
        print(f"Avg prompt length: {df['prompt_length'].mean():.0f} chars")
        
        correlation = df['response_time'].corr(df['satisfaction'])
        print(f"\nResponse time vs Satisfaction correlation: {correlation:.3f}")
        
        return df
    
    def visualize_hci_patterns(self, df):
        """Visualize HCI study results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Satisfaction distribution
        axes[0, 0].hist(
            df['satisfaction'],
            bins=20,
            color='#2196F3',
            alpha=0.8,
            edgecolor='white'
        )
        axes[0, 0].set_title(
            'User Satisfaction Distribution',
            fontweight='bold'
        )
        axes[0, 0].set_xlabel('Satisfaction Score (1-10)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(
            df['satisfaction'].mean(),
            color='red',
            linestyle='--',
            label=f"Mean: {df['satisfaction'].mean():.2f}"
        )
        axes[0, 0].legend()
        
        # Response time vs satisfaction
        axes[0, 1].scatter(
            df['response_time'],
            df['satisfaction'],
            alpha=0.3,
            color='#4CAF50',
            s=20
        )
        z = np.polyfit(
            df['response_time'],
            df['satisfaction'], 1
        )
        p = np.poly1d(z)
        x_line = np.linspace(
            df['response_time'].min(),
            df['response_time'].max(), 100
        )
        axes[0, 1].plot(
            x_line, p(x_line),
            'r--', alpha=0.8,
            label='Trend'
        )
        axes[0, 1].set_title(
            'Response Time vs Satisfaction',
            fontweight='bold'
        )
        axes[0, 1].set_xlabel('Response Time (s)')
        axes[0, 1].set_ylabel('Satisfaction Score')
        axes[0, 1].legend()
        
        # Task completion rate
        completion = df['task_completed'].value_counts()
        axes[1, 0].pie(
            completion.values,
            labels=['Completed', 'Not Completed'],
            colors=['#4CAF50', '#F44336'],
            autopct='%1.1f%%',
            startangle=90
        )
        axes[1, 0].set_title(
            'Task Completion Rate',
            fontweight='bold'
        )
        
        # Interactions per user
        user_interactions = df.groupby('user_id').size()
        axes[1, 1].hist(
            user_interactions.values,
            bins=15,
            color='#FF9800',
            alpha=0.8,
            edgecolor='white'
        )
        axes[1, 1].set_title(
            'Interactions per User',
            fontweight='bold'
        )
        axes[1, 1].set_xlabel('Number of Interactions')
        axes[1, 1].set_ylabel('Number of Users')
        
        plt.suptitle(
            'HCI Study: LLM Interaction Patterns\nAuthor: Jajitha',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig('results/hci_analysis.png', dpi=150)
        plt.show()
        print("HCI analysis visualization saved!")


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    print("="*50)
    print("ADVANCED NLP AND AI - STAGE 5")
    print("Author: Jajitha")
    print("="*50)
    
    # 1. Text Processing
    print("\n1. TEXT PROCESSING PIPELINE")
    processor = TextProcessor()
    
    sample_texts = [
        "GKS scholarship application for Korean university",
        "Natural language processing with transformer models",
        "TOPIK preparation for Korean language proficiency",
        "Research paper on LLMs and human computer interaction",
        "Deep learning for NLP using Python and PyTorch"
    ]
    
    processor.build_vocab(sample_texts)
    
    encoded = processor.encode(
        "GKS scholarship Korean university NLP research"
    )
    print(f"Token count: {encoded['token_count']}")
    print(f"Input IDs (first 10): {encoded['input_ids'][:10]}")
    
    # 2. Self Attention
    print("\n2. SELF-ATTENTION MECHANISM")
    attention = SelfAttention(d_model=64, num_heads=8)
    
    seq_len = 8
    d_model = 64
    x = np.random.randn(seq_len, d_model)
    output, weights = attention.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    tokens = [
        'GKS', 'scholarship', 'Korean',
        'NLP', 'research', 'TOPIK',
        'university', 'AI'
    ]
    attention.visualize_attention(weights, tokens)
    
    # 3. LLM Simulator
    print("\n3. LLM INTERACTION SIMULATOR")
    llm = LLMSimulator("ResearchLLM-v1")
    
    prompts = [
        "What is the GKS scholarship process?",
        "How to prepare for TOPIK Korean exam?",
        "Explain NLP and transformer models",
        "How to write a research paper in NLP?",
        "What is human computer interaction?"
    ]
    
    print("\nSimulating conversations:")
    for prompt in prompts:
        response, tokens, time_ms = llm.generate_response(prompt)
        print(f"\nQ: {prompt}")
        print(f"A: {response[:80]}...")
        print(f"   Tokens: {tokens} | Time: {time_ms:.2f}ms")
    
    stats = llm.get_conversation_stats()
    print("\nConversation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    llm.export_conversation()
    
    # 4. HCI Study
    print("\n4. HCI STUDY ANALYSIS")
    hci = HCIStudyAnalyzer()
    hci.generate_synthetic_study_data(n_users=50)
    df = hci.analyze_patterns()
    hci.visualize_hci_patterns(df)
    
    print("\n" + "="*60)
    print("🎉 ALL 5 STAGES COMPLETE!")
    print("Python Learning Journey: Fundamentals → Advanced AI")
    print("Author: Jajitha")
    print("Destination: Korea 🇰🇷")
    print("="*60)
