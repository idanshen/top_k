import torch
from transformers import LogitsProcessor



class TopKLogitsProcessor(LogitsProcessor, torch.nn.Module):
    """A logits processor that implements dynamic top-k sampling.

    This processor samples a token from the softmax distribution to determine a score threshold,
    then masks out all logits below that threshold. This creates a dynamic top-k effect where
    the k value is determined by the sampled token's score.

    Args:
        temp_1 (float, optional): Temperature parameter for the softmax distribution used in sampling.
            Higher values make the distribution more uniform, lower values make it more peaked.
            Defaults to 1.0.
    """

    def __init__(
        self,
        temp_1: float = 1.0,
    ):
        super().__init__()
        self.temp_1 = temp_1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        This logit processor implements the top-k sampling scheme. 
        It first samples a token from the distribution to determine the top-k threshold, and then masks out all the scores that are not the top-k.
        Args:
            input_ids: (batch_size, seq_len)
            scores: (batch_size, vocab_size)
        """
        distribution = torch.nn.functional.softmax(scores / self.temp_1, dim=-1) # (batch_size, vocab_size)
        sampled_ids = torch.multinomial(distribution, num_samples=1) # (batch_size, 1)
        sampled_scores = torch.gather(scores, 1, sampled_ids) # (batch_size, 1)

        mask = scores < sampled_scores
        processed_scores = scores.masked_fill(mask, -float('inf'))
        return processed_scores