# Deploying to AWS: a beginner's guide

## The words you will meet

- **EC2 instance**: a computer you rent from Amazon and control over the internet. Our app will run on it.
- **Security group**: the instance's firewall. It decides who can reach it and on which port.
- **Key pair (.pem file)**: your password for logging in to the instance. Amazon gives you the file once. Keep it safe and never put it in git.
- **Public IPv4 address**: the instance's internet address. You type it into the browser.
- **SSH**: the way you type commands on the instance from your own PC.

How the pieces fit: the browser talks to **nginx** (port 80), which hands the request to **gunicorn** (port 8000), which runs our Flask app. **systemd** starts gunicorn automatically and restarts it if it crashes. We use ONE gunicorn worker because the app keeps its history in memory.

> The dashboard has **no login**. You will allow port 80 only from your own IP address, so strangers cannot use it.

## Part A. AWS account (skip if you have one)

1. Go to https://aws.amazon.com and choose **Create an AWS account**. It needs an email and a card. Amazon charges the card only for what you use.
2. Sign in to the **AWS Management Console**.
3. Top right, choose the **Region** closest to you (for example Asia Pacific (Mumbai)). Everything you create lives in that region, so use the same one later.
4. **Keep it free.** Search **Billing and Cost Management**, open **Free Tier**, and read what your account includes (new accounts get a free plan with credits for the first months; older accounts get 750 hours per month of a small instance for 12 months). Then search **Budgets**, create a **zero-spend budget** (or a 1 USD monthly budget) with your email, so Amazon warns you the moment anything is charged.

## Part B. Create the instance

1. In the search bar type **EC2** and open it. Click **Launch instance**.
2. **Name**: `ai-resource-manager`.
3. **Application and OS Images**: choose **Ubuntu**, then **Ubuntu Server 24.04 LTS** (64-bit x86).
4. **Instance type**: look for the label **Free tier eligible** next to the type. Pick **t3.small** (2 GB) if it carries that label, otherwise **t3.micro** (1 GB). The setup script adds swap space automatically on 1 GB machines, so t3.micro works, just slower. Do not pick a type without the free-tier label.
5. **Key pair (login)**: click **Create new key pair**. Name it `ai-rm-key`, type **RSA**, format **.pem**, then **Create key pair**. The file downloads (usually to your Downloads folder). Do not lose it.
6. **Network settings**: click **Edit**, then:
   - Firewall: **Create security group**.
   - The SSH rule (port 22): set **Source type** to **My IP**.
   - Click **Add security group rule**: Type **HTTP**, port 80, **Source type: My IP**.
   - Make sure there is no rule that allows `0.0.0.0/0` (anywhere).
7. **Configure storage**: set **20 GiB**, type **gp3** (up to 30 GiB is inside the free allowance).
8. Click **Launch instance**, then **View all instances**.
9. Wait until **Instance state** is **Running** and **Status check** shows **2/2 checks passed** (about 2 minutes).
10. Click the instance and copy its **Public IPv4 address** (for example `13.233.10.20`).

Note: "My IP" is your IP at this moment. If your home or college IP changes later, you will be locked out. Fix it under the instance's **Security** tab: click the security group, **Edit inbound rules**, and set both rules to **My IP** again.

## Part C. Connect from Windows

1. Open **PowerShell**.
2. Allow only you to read the key file, otherwise ssh refuses it (replace `YOURNAME` and the path):
   ```
   icacls "C:\Users\YOURNAME\Downloads\ai-rm-key.pem" /inheritance:r
   icacls "C:\Users\YOURNAME\Downloads\ai-rm-key.pem" /grant:r "$($env:USERNAME):(R)"
   ```
3. Connect (replace the IP with yours):
   ```
   ssh -i "C:\Users\YOURNAME\Downloads\ai-rm-key.pem" ubuntu@13.233.10.20
   ```
4. The first time it asks **Are you sure you want to continue connecting**. Type `yes` and press Enter.
5. When the prompt changes to `ubuntu@ip-...:~$`, you are typing commands on the instance.

(The browser button "EC2 Instance Connect" will NOT work with this firewall, because it connects from Amazon's addresses, not yours.)

## Part D. Install the app

Run these on the instance, one block at a time:

```
git clone https://github.com/Ramnihash/AI-Cloud-Resource-Manager.git
cd AI-Cloud-Resource-Manager
git checkout ramnihash/dashboard
bash deployment/aws/setup.sh
```

`git checkout` is needed until your branch is merged into main.

The script takes roughly 5 to 10 minutes. It installs nginx and Python tools, creates a virtual environment, installs CPU-only PyTorch and the project requirements, trains the device anomaly detector (it prints the "detected 4 of 4" lines), and starts the app as a service. Stay connected until it prints **Done**.

## Part E. Open the dashboard

In your PC's browser go to `http://13.233.10.20/dashboard/` (your IP, plain `http`, not `https`).

Check each page: press **Step System** a few times, run the Memory comparison, and on the Device page pick a fault and press **Scan with detector**.

If the page does not load, run these on the instance:
```
sudo systemctl status ai-resource-manager
curl http://127.0.0.1:8000/health
journalctl -u ai-resource-manager -n 50
```
`{"status":"ok"}` from curl means the app is fine and the problem is the firewall (check that your current IP is in both security group rules).

## Part F. Updating later

After you push new commits from your PC:
```
cd ~/AI-Cloud-Resource-Manager
git pull
bash deployment/aws/update.sh
```
A restart clears the in-memory step history and device actions.

## Part G. Costs and shutting down

- Staying free: use one free-tier-eligible instance, keep the zero-spend budget alert on, and do not create extra instances, Elastic IPs or load balancers. Amazon charges a small hourly fee for every public IPv4 address, which the free plan covers only for your first instance and only within the free period. Check **Billing, Free Tier** now and then to see what is left.
- When you are not using it: EC2, select the instance, **Instance state**, **Stop instance**. A stopped instance stops using compute hours, but its disk still counts toward the free storage allowance. Start it again later; the **Public IPv4 address changes** after a stop, so copy the new one and update the browser link and ssh command.
- To delete everything for good: **Instance state**, **Terminate instance**. Then delete the key pair and the security group if you no longer need them.

## Part H. Real backups to S3 (optional)

By default a "backup" is simulated. To store real backup records in S3:

1. **Create the bucket.** Console, search **S3**, **Create bucket**. Name it something unique like `ai-rm-backups-yourname`, same region as the instance, keep **Block all public access** ON. Create.
2. **Create a role for the instance.** Search **IAM**, **Roles**, **Create role**. Trusted entity: **AWS service**, use case **EC2**. Skip the managed policies for now, name it `ai-rm-ec2-role`, create it. Open the role, **Add permissions**, **Create inline policy**, JSON tab, and paste (replace the bucket name):
   ```
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": ["s3:PutObject"],
       "Resource": "arn:aws:s3:::ai-rm-backups-yourname/*"
     }]
   }
   ```
   Name the policy `ai-rm-s3-write` and create it. The role can only write new objects to that one bucket.
3. **Attach the role to the instance.** EC2, select the instance, **Actions**, **Security**, **Modify IAM role**, choose `ai-rm-ec2-role`, **Update IAM role**. No keys are needed on the machine.
4. **Tell the app which bucket to use.** On the instance:
   ```
   echo 'BACKUP_S3_BUCKET=ai-rm-backups-yourname' | sudo tee /etc/ai-resource-manager.env
   echo 'AWS_DEFAULT_REGION=ap-south-1' | sudo tee -a /etc/ai-resource-manager.env
   cd ~/AI-Cloud-Resource-Manager && git pull && bash deployment/aws/update.sh
   ```
   Use your own bucket name and region code.
5. **Test.** On the Device page run an action with probability 0.85 and a backup device. The Cloud backup card should say **S3** and show an `s3://...` location. In the S3 console, open the bucket and look under `device-backups/`.

If the card says "Last backup failed", the message in the Recent actions table says why (usually the role is not attached or the bucket name is wrong). S3 storage for these tiny files is free within the free tier.

## Part I. Password for demos (Basic Auth)

The dashboard has no login. For a demo with teammates, put one shared password in front of it:

```
cd ~/AI-Cloud-Resource-Manager
bash deployment/aws/basic-auth.sh on
```

Type a username and a password (nothing shows while you type the password). The browser then asks for them before showing any page. The password is stored only on the server, hashed. Turn it off with `bash deployment/aws/basic-auth.sh off`.

Then, in the AWS Console, EC2, your instance, **Security**, the security group, **Edit inbound rules**: change the HTTP (port 80) source to your teammates' IPs, or to **Anywhere-IPv4** for a short demo. Set it back to **My IP** afterwards. Keep SSH (port 22) on **My IP** always.

Plain HTTP still sends the password unencrypted, so use a password you use nowhere else, and only for demos.

## Part J. Extra protection when the site is open to everyone

```
cd ~/AI-Cloud-Resource-Manager
git pull
bash deployment/aws/harden.sh
```

This limits how fast one visitor can hit the site, hides the nginx version, adds basic security headers, and installs fail2ban, which bans an IP for an hour after 5 wrong passwords (website login or ssh). Run it after Basic Auth is on.

## Not covered yet

- HTTPS (needs a domain name). Keep the firewall restricted when no password is set.
