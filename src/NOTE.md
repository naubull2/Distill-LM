# Choosing Which Attention Heads to Prune

Choosing attention heads to prune can be based on static rules, heuristics, or trainable methods:

## Static or Heuristic-Based Pruning

These methods don't involve additional training but rely on patterns observed in attention.

*Rule of Thumb:*
Prune heads that contribute the least to the final output. Studies (e.g., by Michel et al., Are Sixteen Heads Really Better than One?) suggest many heads are redundant.

- Start with heads that show:
  - `Low attention entropy`: Heads focusing narrowly on certain tokens.
  - `Weak correlation`: Heads whose outputs have little impact on subsequent layers.
