import math
import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_pages: int, d_model: int, nhead: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(num_pages, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.linear = nn.Linear(d_model, num_pages)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len] containing page IDs.

        Returns:
            Tensor of shape [batch_size, num_pages] containing logits for the next page.
        """

        src = self.embedding(src)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)


        last_hidden = output[:, -1, :]

        logits = self.linear(last_hidden)
        return logits


class TransformerPagePredictor:
    """Transformer-based page predictor for Module 2.

    Learns to predict the next page from a context window of previous pages using
    a single-layer Transformer encoder.

    Lifecycle::

        predictor = TransformerPagePredictor(num_pages=32)
        predictor.train(sequence)          # required before predict/evaluate
        predictor.predict(context)
        predictor.evaluate(test_sequence)
    """

    def __init__(
        self,
        num_pages: int,
        context_len: int = 4,
        d_model: int = 32,
        nhead: int = 2,
        d_ff: int = 64,
        seed: int = 42,
        dropout: float = 0.0,
    ):
        if not isinstance(num_pages, int) or isinstance(num_pages, bool) or num_pages < 1:
            raise ValueError(f"num_pages must be an integer >= 1, got {num_pages!r}")
        if not isinstance(context_len, int) or isinstance(context_len, bool) or context_len < 1:
            raise ValueError(f"context_len must be an integer >= 1, got {context_len!r}")

        self.num_pages = num_pages
        self.context_len = context_len
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.seed = seed
        self.dropout = dropout

        self._trained = False
        self.model = None

    def _init_model(self):
        torch.manual_seed(self.seed)
        self.model = TransformerModel(
            num_pages=self.num_pages,
            d_model=self.d_model,
            nhead=self.nhead,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )

    def train(self, sequence: list, epochs: int = 5, lr: float = 1e-3) -> dict:
        """Learn from a training sequence using a sliding window approach.

        Args:
            sequence: List of page IDs.
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Dictionary with training metrics.
        """
        if not isinstance(sequence, list):
            raise TypeError(f"sequence must be a list, got {type(sequence).__name__}")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError(f"epochs must be an integer >= 1, got {epochs!r}")
        if not isinstance(lr, (float, int)) or isinstance(lr, bool) or lr <= 0:
            raise ValueError(f"lr must be a positive float, got {lr!r}")

        if len(sequence) < self.context_len + 1:
            raise ValueError(
                f"sequence must have at least context_len + 1 ({self.context_len + 1}) "
                f"elements, got {len(sequence)}"
            )


        for p in sequence:
            if not (0 <= p < self.num_pages):
                raise ValueError(f"Page ID {p} in sequence is out of bounds [0, {self.num_pages - 1}]")

        self._init_model()
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)


        X = []
        Y = []
        for i in range(len(sequence) - self.context_len):
            X.append(sequence[i : i + self.context_len])
            Y.append(sequence[i + self.context_len])

        X_tensor = torch.tensor(X, dtype=torch.long)
        Y_tensor = torch.tensor(Y, dtype=torch.long)

        final_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.model(X_tensor)
            loss = criterion(logits, Y_tensor)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self._trained = True

        return {
            "epochs": epochs,
            "final_loss": final_loss,
            "num_pages": self.num_pages,
            "context_len": self.context_len,
            "training_samples": len(X),
        }

    def predict(self, context: list, top_k: int = 1) -> list:
        """Predict the next page given a context window of length context_len.

        Args:
            context: List of exactly context_len page IDs.
            top_k: Number of candidates to return.

        Returns:
            List of dicts: {"page": int, "probability": float} sorted descending.
        """
        self._require_trained("predict")

        if not isinstance(context, list):
            raise TypeError(f"context must be a list, got {type(context).__name__}")
        if len(context) != self.context_len:
            raise ValueError(f"context must have length {self.context_len}, got {len(context)}")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k!r}")

        for p in context:
            if not isinstance(p, int) or isinstance(p, bool):
                raise TypeError(f"Context elements must be integers, got {type(p).__name__}")
            if not (0 <= p < self.num_pages):
                raise ValueError(f"Page ID {p} in context is out of bounds [0, {self.num_pages - 1}]")

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor([context], dtype=torch.long)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)


            k = min(top_k, self.num_pages)
            top_probs, top_indices = torch.topk(probs, k)

            candidates = []
            for prob, idx in zip(top_probs, top_indices):
                candidates.append({
                    "page": idx.item(),
                    "probability": prob.item()
                })

        return candidates

    def evaluate(self, test_sequence: list, top_k: int = 1) -> dict:
        """Evaluate prediction accuracy on a test sequence.

        Args:
            test_sequence: List of page IDs.
            top_k: Top-K accuracy threshold.

        Returns:
            Dictionary with evaluation metrics.
        """
        self._require_trained("evaluate")

        if not isinstance(test_sequence, list):
            raise TypeError(f"test_sequence must be a list, got {type(test_sequence).__name__}")
        if len(test_sequence) < self.context_len + 1:
            raise ValueError(
                f"test_sequence must have at least context_len + 1 ({self.context_len + 1}) "
                f"elements, got {len(test_sequence)}"
            )
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k!r}")

        total = len(test_sequence) - self.context_len
        correct_top1 = 0
        correct_topk = 0

        for i in range(total):
            context = test_sequence[i : i + self.context_len]
            actual_next = test_sequence[i + self.context_len]

            preds = self.predict(context, top_k=top_k)
            pred_pages = [p["page"] for p in preds]

            if pred_pages and pred_pages[0] == actual_next:
                correct_top1 += 1
            if actual_next in pred_pages:
                correct_topk += 1

        top1_acc = correct_top1 / total if total > 0 else 0.0
        topk_acc = correct_topk / total if total > 0 else 0.0

        return {
            "total_predictions": total,
            "correct_top1": correct_top1,
            "correct_topk": correct_topk,
            "top1_accuracy": top1_acc,
            "topk_accuracy": topk_acc,
            "top_k": top_k,
        }

    def _require_trained(self, method_name: str) -> None:
        """Raise RuntimeError if train() has not yet been called."""
        if not self._trained:
            raise RuntimeError(
                f"TransformerPagePredictor.{method_name}() called before train(). "
                f"Call train(sequence) first."
            )
