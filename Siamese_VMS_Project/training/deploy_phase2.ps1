$ip = "52.207.32.24"
$key = "C:\Users\FAWWAD\Downloads\MyKey.pem"
$user = "ec2-user"
$remote = "~/Siamese_VMS_Project"

Write-Host "Copying Phase-2 files to AWS..."
scp -i $key -o StrictHostKeyChecking=no `
    ../core/augment_utils.py tts_bank.py dataset_v2.py train_siamese_v2.py ../core/siamese_model.py `
    ${user}@${ip}:${remote}/

Write-Host "Launching TTS bank generation + GRL training (chained, background)..."
ssh -i $key -o StrictHostKeyChecking=no ${user}@${ip} `
    "cd ~/Siamese_VMS_Project && nohup bash -c 'python3 tts_bank.py && python3 -u train_siamese_v2.py' > phase2.log 2>&1 < /dev/null & echo 'Started, PID:' `$!"

Write-Host "=========================================================="
Write-Host "Phase 2 launched. Monitor with:"
Write-Host "  ssh -i $key ${user}@${ip} 'tail -f ~/Siamese_VMS_Project/phase2.log'"
Write-Host "Stages: (1) TTS bank ~1650 words x 5 voices on the L4 GPU,"
Write-Host "        (2) 40-epoch GRL training with cross-domain AUC eval."
Write-Host "=========================================================="
