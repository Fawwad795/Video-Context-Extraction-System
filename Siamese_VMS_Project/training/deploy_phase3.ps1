$ip = "54.167.195.198"
$key = "C:\Users\FAWWAD\Downloads\MyKey.pem"
$user = "ec2-user"
$remote = "~/Siamese_VMS_Project"

Write-Host "Copying Phase-3 files to AWS..."
scp -i $key -o StrictHostKeyChecking=no `
    ../core/embedders.py ../core/augment_utils.py dataset_v3.py train_siamese_v3.py `
    ${user}@${ip}:${remote}/

Write-Host "Launching feature-cache build + attentive-head training (chained, background)..."
ssh -i $key -o StrictHostKeyChecking=no ${user}@${ip} `
    "cd ~/Siamese_VMS_Project && pip3 install -q g2p_en 2>/dev/null; nohup bash -c 'python3 -u dataset_v3.py --build-cache && python3 -u train_siamese_v3.py' > phase3_head.log 2>&1 < /dev/null & echo 'Started, PID:' `$!"

Write-Host "=========================================================="
Write-Host "Phase 3 launched. Monitor with:"
Write-Host "  ssh -i $key ${user}@${ip} 'tail -f ~/Siamese_VMS_Project/phase3_head.log'"
Write-Host "Stages: (1) WavLM-L10 feature cache (~1000 MSWC classes + TTS bank),"
Write-Host "        (2) 40-epoch sub-center ArcFace training of the attentive head."
Write-Host "Fetch the checkpoint afterwards:"
Write-Host "  scp -i $key ${user}@${ip}:~/Siamese_VMS_Project/siamese_v3_best.pth ../checkpoints/"
Write-Host "=========================================================="
