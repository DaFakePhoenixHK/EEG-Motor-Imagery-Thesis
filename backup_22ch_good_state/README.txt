Backup: 22ch EEGNet "good" state (after the 4 fixes)
====================================================
- train_eval_acc / train_eval_loss (eval mode on train set)
- Cross-subject: NO trial-level re-split (Train=train_subjects only, Val=val_subjects only, Test=test_subjects only)
- val_loss_weighted + val_loss_unweighted logged
- Class weights only (no oversampling)

TO RESTORE if you mess up the code later:
1. Copy scripts/train_eegnet_22ch.py from this folder over your scripts/train_eegnet_22ch.py
2. Copy src/plotting.py from this folder over your src/plotting.py

Or from command line (run from project root, e.g. C:\Users\User\Desktop\Thesis\files):
  copy /Y backup_22ch_good_state\train_eegnet_22ch.py scripts\
  copy /Y backup_22ch_good_state\plotting.py src\
