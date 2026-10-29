# experiments/unlearn/sweeps/grids.psd1
# ======================================
# Sweep grids as data, consumed by sweep.ps1.
#
# Each entry is  "<dataset>/<method>" = @{ primary = @{ flag; values }; epochs }
# The sweep is primary.values x epochs.
#
# Ranges below reflect MEASURED behaviour, not guesses -- see the notes on each.

@{

    # -----------------------------------------------------------------------
    # dSprites  (baseline retain val loss = 36.9)
    # -----------------------------------------------------------------------

    # Optuna (40 trials, held-out scored) put the usable front at
    # lambda 2.5e-4 .. 7e-4; collapse is hard above ~1e-3 (retain ELBO > 6000).
    "dsprites/neg_grad" = @{
        primary = @{ flag = "--lambda-forget"
                     values = @("0.00015", "0.00025", "0.0003", "0.0005", "0.0007") }
        epochs  = @("5", "13", "20")
    }

    # Finetune needs far more epochs at far lower lambda; near-oracle result
    # was found around lam=0.0001 ep=90.
    "dsprites/finetune" = @{
        primary = @{ flag = "--lambda-forget"
                     values = @("0.00005", "0.0001", "0.0002", "0.0003") }
        epochs  = @("30", "60", "90")
    }

    # Measured at 3 epochs after the loss-scale fix:
    #   alpha=1  -> retain val  50.1     alpha=10 -> retain val 192.9
    #   alpha=10 with --elbo-anchor 1 -> retain val 61.0 at equal forget pressure.
    # The old grid ran to alpha=50, which is far past anything useful.
    "dsprites/feature_unlearn" = @{
        primary = @{ flag = "--alpha"; values = @("0.5", "1", "2", "5", "10") }
        epochs  = @("5", "10", "20")
    }

    # RMU: --alpha (retain anchor) is the sensitive knob, so it is the sweep axis.
    # Measured at c=1, 1 epoch (baseline val 36.9):
    #     alpha=1   -> val 1781 (destroyed)      alpha=100  -> val  37.5 (no forgetting)
    #     alpha=10  -> val  128                  alpha=1000 -> val  35.8 (no forgetting)
    # The usable trade-off is 10-100, so the grid concentrates there.
    # --c (steering magnitude) is left at its default of 1.0; baseline mean
    # ||mu|| is ~2.5, so c=1 places the target well inside the latent cloud.
    # To sweep c instead, pass -Extra "--c <v>" -- it is recorded in the run
    # directory name, so runs will not collide.
    "dsprites/rmu" = @{
        primary = @{ flag = "--alpha"; values = @("10", "20", "35", "50", "75", "100") }
        epochs  = @("5", "10", "20")
    }

    # -----------------------------------------------------------------------
    # CelebA  (baseline retain val loss = 379)
    # -----------------------------------------------------------------------

    # lam=0.01 ep=40 passed the retain threshold; collapse observed at lam=0.02.
    "celeba/neg_grad" = @{
        primary = @{ flag = "--lambda-forget"; values = @("0.005", "0.01", "0.015") }
        epochs  = @("20", "40", "60")
    }

    "celeba/finetune" = @{
        primary = @{ flag = "--lambda-forget"; values = @("0.005", "0.01", "0.02") }
        epochs  = @("15", "30", "45")
    }

    "celeba/feature_unlearn" = @{
        primary = @{ flag = "--alpha"; values = @("0.5", "1", "2", "5", "10") }
        epochs  = @("5", "10", "20")
    }

    # Same reasoning as dsprites/rmu -- alpha is the sweep axis.  CelebA's
    # trade-off point has not been calibrated yet; widen if all runs collapse
    # or all show no forgetting.
    "celeba/rmu" = @{
        primary = @{ flag = "--alpha"; values = @("10", "25", "50", "100") }
        epochs  = @("5", "10", "20")
    }
}
