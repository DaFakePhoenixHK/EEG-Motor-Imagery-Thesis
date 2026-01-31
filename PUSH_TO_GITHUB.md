# How to push this project to GitHub

Your repo: **https://github.com/DaFakePhoenixHK/EEG-Motor-Imagery-Thesis**

---

## Step 1: Open terminal in the project folder

- In VS Code / Cursor: **Terminal → New Terminal** (or `` Ctrl+` ``)
- Or open **PowerShell** / **Command Prompt** and run:
  ```text
  cd C:\Users\User\Desktop\Thesis\files
  ```

---

## Step 2: Make sure everything is committed

If you added or changed files after the first commit, run:

```powershell
git add -A
git status
git commit -m "Add scripts and src"
```

(Only run `git commit` if `git status` shows files to commit.)

---

## Step 3: Add your GitHub repo as “origin”

One-time setup:

```powershell
git remote add origin https://github.com/DaFakePhoenixHK/EEG-Motor-Imagery-Thesis.git
```

If you see “remote origin already exists”, use:

```powershell
git remote set-url origin https://github.com/DaFakePhoenixHK/EEG-Motor-Imagery-Thesis.git
```

---

## Step 4: Push to GitHub

First time (creates `main` on GitHub and pushes):

```powershell
git branch -M main
git push -u origin main
```

If your branch is already called `master` and you want to keep it:

```powershell
git push -u origin master
```

- **GitHub will ask you to sign in** (browser or username/password/token).
- If you use **2FA**, use a **Personal Access Token** instead of your password:  
  GitHub → **Settings → Developer settings → Personal access tokens** → create a token with `repo` scope, then paste it when Git asks for a password.

---

## Step 5: Check on GitHub

Open: **https://github.com/DaFakePhoenixHK/EEG-Motor-Imagery-Thesis**

You should see your code there.

---

## Later: after you change code

```powershell
cd C:\Users\User\Desktop\Thesis\files
git add -A
git commit -m "Describe what you changed"
git push
```

---

## If something goes wrong

- **“Permission denied” / “index.lock”**  
  Close other programs that might use Git (IDE, other terminals), then delete the lock file and try again:
  ```powershell
  Remove-Item -Force .git\index.lock -ErrorAction SilentlyContinue
  git add -A
  git commit -m "Your message"
  ```

- **“Authentication failed”**  
  Use a Personal Access Token as the password, or set up SSH and use the SSH URL:
  ```text
  git@github.com:DaFakePhoenixHK/EEG-Motor-Imagery-Thesis.git
  ```

- **“scripts/ or src/ not in repo”**  
  Make sure you’re in `C:\Users\User\Desktop\Thesis\files` and run:
  ```powershell
  git add scripts/ src/
  git status
  git commit -m "Add scripts and src"
  git push
  ```
