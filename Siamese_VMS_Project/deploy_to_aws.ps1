$ip = "13.220.224.113"
$key = "C:\Users\FAWWAD\Downloads\MyKey.pem"
$user = "ec2-user"

Write-Host "Creating project directory on AWS..."
ssh -i $key -o StrictHostKeyChecking=no $user@$ip "mkdir -p ~/Siamese_VMS_Project"

Write-Host "Copying files to AWS..."
scp -i $key -o StrictHostKeyChecking=no siamese_model.py dataset.py train_siamese.py ${user}@${ip}:~/Siamese_VMS_Project/

Write-Host "Starting Background Training on AWS (this will take several hours)..."
ssh -i $key -o StrictHostKeyChecking=no $user@$ip "cd ~/Siamese_VMS_Project && pip install transformers librosa 'datasets<3.0.0' soundfile && nohup python train_siamese.py > training.log 2>&1 < /dev/null &"

Write-Host "=========================================================="
Write-Host "Deployment triggered successfully!"
Write-Host "The 50-epoch training loop is now running in the background."
Write-Host "To monitor progress, you can SSH into your instance and run:"
Write-Host "tail -f ~/Siamese_VMS_Project/training.log"
Write-Host "=========================================================="
