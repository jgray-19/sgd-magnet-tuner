import math


class LRScheduler:
    """
    Cosine-decay learning-rate scheduler with cosine warmup.

    Usage:
        scheduler = LRScheduler(
            warmup_epochs=2000,
            decay_epochs=3000,
            start_lr=1e-6,
            max_lr=1e-4,
            min_lr=1e-5,
        )
        lr = scheduler(epoch)
    """

    def __init__(
        self,
        warmup_epochs: int,
        decay_epochs: int,
        start_lr: float,
        max_lr: float,
        min_lr: float,
    ):
        """
        Initialise the scheduler.

        Args:
            warmup_epochs: Number of epochs for cosine warmup.
            decay_epochs: Number of epochs to decay LR after warmup.
            start_lr: Learning rate at epoch=0.
            max_lr: Peak learning rate after warmup.
            min_lr: Final learning rate after decay.
        """
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr

    def __call__(self, epoch: int) -> float:
        """
        Compute the learning rate for a given epoch index (0-based).

        Args:
            epoch: Current epoch (0-based).

        Returns:
            Learning rate for this epoch.
        """
        # Shift to 1-based epoch index.
        e = epoch + 1

        # Cosine warmup phase: Cosine increase from start_lr to max_lr.
        if e <= self.warmup_epochs:
            if self.warmup_epochs == 1:
                return self.max_lr
            factor = (1 - math.cos(math.pi * (e - 1) / (self.warmup_epochs - 1))) / 2
            return self.start_lr + (self.max_lr - self.start_lr) * factor

        # Decay phase: Cosine decay from max_lr to min_lr over decay_epochs.
        elif e <= self.warmup_epochs + self.decay_epochs:
            progress = (e - self.warmup_epochs) / self.decay_epochs
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine

        # After decay phase: Learning rate stays at min_lr.
        else:
            return self.min_lr

